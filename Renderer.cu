#include "Hittable/Sphere.h"
#include "Renderer.h"
#include "CommonMath.h"
#include <cstdio>
inline __device__ void writeColor(uchar8* pixelPtr, const Color& color, int samplesPerPixel) {
    float scale = 1.0f / samplesPerPixel;

    pixelPtr[0] = (uchar8) (256 * clamp(color[0] * scale, 0.0f, 0.999f));
    pixelPtr[1] = (uchar8) (256 * clamp(color[1] * scale, 0.0f, 0.999f));
    pixelPtr[2] = (uchar8) (256 * clamp(color[2] * scale, 0.0f, 0.999f));
}

__global__ void sampleRender(uchar8* colorData, const Camera* cam,
                             Hittable** world, int samplesPerPixel) {
    extern __shared__ Color pixelColor[];
    (*pixelColor)[0] = (*pixelColor)[1] = (*pixelColor)[2] = 0.0f;

    __syncthreads();

    unsigned int i = blockIdx.x;
    unsigned int j = blockIdx.y;

    float u = i / (float) (gridDim.x - 1);
    float v = (gridDim.y - j - 1) / (float) (gridDim.y - 1);
    Ray ray = cam->getRay(u, v);
    HitRecord record;

    Color currSample;
    if ((*world)->hit(ray, 0, infinity, record)) {
        currSample = 0.5f * (record.normal + Color(1.0f, 1.0f, 1.0f));

    } else {
        Vector3D unitDir = ray.direction().normalized();
        float t = 0.5f * (unitDir.y() + 1.0f);
        currSample = (1.0f - t) * Color(1.0f, 1.0f, 1.0f) + t * Color(0.5f, 0.7f, 1.0f);

    }
   pixelColor->atomicAddVec(currSample);

    __syncthreads();

    if (threadIdx.x == 0) {
        uchar8* pixelPtr = colorData + 3 * (j * gridDim.x + i);
        writeColor(pixelPtr, (*pixelColor), samplesPerPixel);
    }
}

__global__ void createWorld(Hittable** world) {
    Hittable* sphere = new Sphere(Vector3D(0.0f, 0.0f, -1.0f), 0.5f);
    *world = sphere;
}

__global__ void destroyWorld(Hittable** world) {
    delete *world;
}

Renderer::Renderer(int imgWidth, int imgHeight, int samplesPerPixel) {
    mImgWidth = imgWidth;
    mImgHeight = imgHeight;
    mSamplesPerPixel = samplesPerPixel;

    size_t colorBuffSize = 3 * imgWidth * imgHeight;
    cudaMalloc(&mColorBuff_d, colorBuffSize * sizeof(uchar8));
    mColorBuff_h = new uchar8[colorBuffSize * sizeof(uchar8)];

    cudaMalloc(&mWorld_d, sizeof(Hittable*));
    createWorld<<<1, 1>>>(mWorld_d);
}

uchar8* Renderer::render(const Camera* camera) {
    dim3 gridDim(mImgWidth, mImgHeight);
    sampleRender<<<gridDim, mSamplesPerPixel, sizeof(Color)>>>(mColorBuff_d, camera, mWorld_d, mSamplesPerPixel);
    cudaPeekAtLastError();
    size_t colorBuffSize = 3 * mImgWidth * mImgHeight;
    cudaMemcpy(mColorBuff_h, mColorBuff_d, colorBuffSize * sizeof(char), cudaMemcpyDeviceToHost);
    return mColorBuff_h;
}

Renderer::~Renderer() {
    destroyWorld<<<1, 1>>>(mWorld_d);
    cudaFree(mWorld_d);
    cudaFree(mColorBuff_d);
    cudaFree(mColorBuff_h);
}
