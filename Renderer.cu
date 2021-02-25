#include "Vector3D.h"
#include "Hittable/Sphere.h"
#include "Renderer.h"
#include <cstdio>
#include <QDebug>
#include <cfloat>
#include <ctime>



//TODO (1): Написать __global__ функцию для запуска GPU, которая бы производила рендер изображения
// In: Hittable* - указатель на "мир" объектов
// In: Camera* - экземпляр класса камеры для создания лучей
// In, Out: Color* - Массив, заполенный нулевыми значениями, для хранения цветов пикселей
// В одном блоке потоков создаются все лучи, отвечающие за рендер одного пикселя,
// результат каждого суммируется в соотвуствующую ячейку массива при помощи Vector3D::atomicAdd().
// Все входные параметры обязательно(!) должны быть алоцированны на памяти видеокарты, иначе разыменование данных указателей будет невозможно.


inline __device__ float clamp(float x, float min, float max) {
    if (x < min) {
        return min;
    }
    if (x > max) {
        return max;
    }
    return x;
}

inline __device__ void mapToImage(const int index, int& i, int& j, const int height, const int width) {
    i = index / width;
    j = index - i * width;
    i = height - i - 1;
}

__device__ Color backgrColor(const Ray& ray) {
    Vector3D unitDir = ray.direction().normalized();
    float t = 0.5 * (unitDir.y() + 1);
    return (1.0 - t) * Color(1, 1, 1) + t * Color(0.5, 0.7, 1.0);
}

inline __device__ void writeColor(const Color& color, unsigned char* buf, const int index, const int samples) {
    float scale = 1.f / samples;
    buf[3 * index] = (unsigned char) (256 * clamp(color[0] * scale, 0, 0.999));
    buf[3 * index + 1] = (unsigned char) (256 * clamp(color[1] * scale, 0, 0.999));
    buf[3 * index + 2] = (unsigned char) (256 * clamp(color[2] * scale, 0, 0.999));
}

// have to copy image and camera
__global__ void getFastPicture(Image image, int n, const Camera cam, Hittable** world_ptr) {
    Hittable* world = *world_ptr;
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int i, j;
    if (index < n) {
        mapToImage(index, i, j, image.height, image.width);
        float u = (float) j / (image.width - 1);
        float v = (float) i / (image.height - 1);
        Ray ray = cam.getRay(u, v);
        HitRecord hr;
        Color pixColor;
        if (world->hit(ray, 0, FLT_MAX, hr)) { // std::numeric_limits<float>::max() doesn't work(
            pixColor = hr.normal;
        }
        else {
            pixColor = backgrColor(ray);
        }
        //writeColor(pixColor, image.d_pixels, index);
        // that will be faster
        image.d_pixels[3 * index] = (unsigned char) (255.999 * pixColor[0]);
        image.d_pixels[3 * index + 1] = (unsigned char) (255.999 * pixColor[1]);
        image.d_pixels[3 * index + 2] = (unsigned char) (255.999 * pixColor[2]);
    }
}


__global__ void getFinePicture(Image image, int n, const Camera cam, Hittable** world_ptr, curandState* randState) {
    __shared__ Color pixColor; // warning: dynamic initialization is not supported for a function-scope static __shared__ variable within a __device__/__global__ function
    // so I made this shit
    pixColor[0] = pixColor[1] = pixColor[2] = 0;
    __syncthreads();
    Hittable* world = *world_ptr;
    int thrIdx = threadIdx.x /*+ blockIdx.x * blockDim.x*/;
    int picIdx = blockIdx.x;
    int i, j;
    mapToImage(picIdx, i, j, image.height, image.width);
    float u = (float) (j + curand_uniform(&randState[thrIdx])) / (image.width - 1); // (0, 1] distribution(((
    float v = (float) (i + curand_uniform(&randState[thrIdx])) / (image.height - 1);
    Ray ray = cam.getRay(u, v);
    HitRecord hr;
    if (world->hit(ray, 0, FLT_MAX, hr)) { // std::numeric_limits<float>::max() doesn't work(
        pixColor.atomAdd(hr.normal);
    }
    else {
        pixColor.atomAdd(backgrColor(ray));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        writeColor(pixColor, image.d_pixels, picIdx, blockDim.x);
    }
}

unsigned char* Renderer::render() {
    if (!image.pixels) {
        return nullptr;
    }
    cudaError_t error;
    int blocks;
    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    if (mode == RenderMode::FAST) {
        blocks = (image.height * image.width + threadsFast - 1) / threadsFast;
        getFastPicture <<<blocks, threadsFast>>> (image, image.width * image.height, cam, d_world);
    }
    else if (mode == RenderMode::ANTIALIASING) {
        blocks = image.height * image.width;
        getFinePicture <<<blocks, samplesPerPix>>> (image, image.width * image.height, cam, d_world, randState);
    }
    error = cudaMemcpy(image.pixels, image.d_pixels, image.height * image.width * 3 * sizeof(char), cudaMemcpyDeviceToHost);
    cudaEventRecord (stop, 0);
    cudaEventSynchronize (stop);
    cudaEventElapsedTime (&gpuTime, start, stop);
    qDebug() << "cuda memcpy pixels: " << cudaGetErrorString(error);
    std::cout << "render time: " << gpuTime << " ms." << std::endl;
    return image.pixels;
}

__global__ void initWorld(Hittable** world) {
    Hittable* sphere = new Sphere(Vector3D(-0.5,0, -1), 0.5);
    World* res = new World();
    res->add(sphere);
    sphere = new Sphere(Vector3D(0.5,0, -1), 0.5);
    res->add(sphere);
    *world = res;
}

//todo: think how to generate rand numbers and not use 4gb of vram
__global__ void setupKernel(curandState *state, int seed) {
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

// todo: make safer (not only log cudaMalloc errors)
Renderer::Renderer(const int height, const int width):cam((float)width / height) {
    cudaError_t error;
    error = cudaMalloc((void**)&image.d_pixels, height * width * 3 * sizeof(char));
    qDebug() << "cuda malloc pixels: " << cudaGetErrorString(error);
    image.pixels = new unsigned char[height * width * 3];
    if (!image.pixels || !image.d_pixels) {
        qDebug() << "Renderer wasn't created";
        delete[] image.pixels;
        image.pixels = nullptr;
        cudaFree(image.d_pixels);
        image.d_pixels = nullptr;
    }
    image.width = width;
    image.height = height;
    error = cudaMalloc((void**)&d_world, sizeof(Hittable**));
    qDebug() << "cuda malloc world ptr: " << cudaGetErrorString(error);
    initWorld<<<1,1>>>(d_world);
    if (!d_world) {
        qDebug() << "troubles with world creation on the device";
    }
    mode = RenderMode::FAST;
    error = cudaMalloc((void**)&randState, /*height * width * */ samplesPerPix * sizeof(curandState)); // todo: blya kak mnogo
    qDebug() << "cuda malloc rand state: " << cudaGetErrorString(error);
    int seed = clock() % 512;
    setupKernel<<</*image.height * image.width*/1, samplesPerPix>>>(randState, seed);
    cudaDeviceSynchronize();
}

__global__ void uninitWorld(Hittable** world) {
    delete *world;
}

void Renderer::changeMode() {
    if (mode == RenderMode::FAST) {
        mode = RenderMode::ANTIALIASING;
    }
    else if (mode == RenderMode::ANTIALIASING) {
        mode = RenderMode::FAST;
    }
}

Renderer::~Renderer() {
    cudaFree(image.d_pixels);
    uninitWorld<<<1,1>>>(d_world); // delete on the device side
    delete[] image.pixels;
    cudaDeviceSynchronize(); // comment up
    cudaFree(d_world);
    cudaFree(randState);
    qDebug() << "rendered destructed";
}
