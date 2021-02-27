#include "Hittable/Sphere.h"
#include "Renderer.h"
#include "Common/Math.h"
#include <iostream>

inline __device__ void writeColor(uchar8* pixelPtr, const Color& color, int samplesPerPixel) {
    float scale = 1.0f / samplesPerPixel;

    pixelPtr[0] = (uchar8) (256 * clamp(color[0] * scale, 0.0f, 0.999f));
    pixelPtr[1] = (uchar8) (256 * clamp(color[1] * scale, 0.0f, 0.999f));
    pixelPtr[2] = (uchar8) (256 * clamp(color[2] * scale, 0.0f, 0.999f));
}

__device__ Color getColor(const Ray& ray, Hittable* const* world) {
    HitRecord record;

    Color color;
    if ((*world)->hit(ray, 0, infinity, record)) {
        color = 0.5f * (record.normal + Color(1.0f, 1.0f, 1.0f));

    } else {
        Vector3D unitDir = ray.direction().normalized();
        float t = 0.5f * (unitDir.y() + 1.0f);
        color = (1.0f - t) * Color(1.0f, 1.0f, 1.0f) + t * Color(0.5f, 0.7f, 1.0f);

    }
    return color;
}

__global__ void sampleRender(int imgWidth, int imgHeight, int samplesPerPixel,
                             uchar8* colorData,
                             const Camera* cam, Hittable* const* world) {

    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= imgWidth || y >= imgHeight) { return; }

    Color pixelColor;

    for (int i = 0; i < samplesPerPixel; i++) {
        float u = x / (float) (imgWidth - 1);
        float v = (imgHeight - y - 1) / (float) (imgHeight - 1);

        Ray ray = cam->getRay(u, v);
        pixelColor += getColor(ray, world);
    }

    uchar8* pixelPtr = colorData + 3 * (imgWidth * y + x);
    writeColor(pixelPtr, pixelColor, samplesPerPixel);
}

__global__ void createWorld(Hittable** world) {
    Hittable* sphere = new Sphere(Vector3D(0.0f, 0.0f, -1.0f), 0.5f);
    *world = sphere;
}

__global__ void destroyWorld(Hittable** world) {
    delete *world;
}

Renderer::Renderer(const RenderInfo& renderInfo) {
    mImgWidth = renderInfo.imgWidth;
    mImgHeight = renderInfo.imgHeight;
    mSamplesPerPixel = renderInfo.samplesPerPixel;

    mThreadBlockWidth = renderInfo.threadBlockWidth;
    mThreadBlockHeight = renderInfo.threadBlockHeight;

    mColorDataSize = 3 * mImgWidth * mImgHeight;
    cudaMalloc(&mColorData_d, mColorDataSize * sizeof(uchar8));
    mColorData_h = new uchar8[mColorDataSize];

    cudaMalloc(&mWorld_d, sizeof(Hittable*));
    createWorld<<<1, 1>>>(mWorld_d);
}

uchar8* Renderer::render(const Camera* camera) {
    clock_t start, stop;
    start = clock();

    int gridWidth = (mImgWidth + mThreadBlockWidth - 1) / mThreadBlockWidth;
    int gridHeight = (mImgHeight + mThreadBlockHeight - 1) / mThreadBlockHeight;

    dim3 gridDim(gridWidth, gridHeight);
    dim3 blockDim(mThreadBlockWidth, mThreadBlockHeight);

    sampleRender<<<gridDim, blockDim>>>(mImgWidth, mImgHeight, mSamplesPerPixel,
                                        mColorData_d,
                                        camera, mWorld_d);

    cudaMemcpy(mColorData_h, mColorData_d, mColorDataSize * sizeof(uchar8), cudaMemcpyDeviceToHost);

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cout << "took " << timer_seconds << " seconds.\n";

    return mColorData_h;
}

Renderer::~Renderer() {
    destroyWorld<<<1, 1>>>(mWorld_d);

    cudaFree(mWorld_d);
    cudaFree(mColorData_d);

    delete mColorData_h;
}
