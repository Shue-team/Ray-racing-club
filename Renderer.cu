#include "Renderer.h"

#include "Hittable/Sphere.h"
#include "Hittable/HittableList.h"

#include "Common/Math.h"
#include "Common/Rand.h"

#include "Material/Lambertian.h"

#include <iostream>

__device__ void writeColor(uchar8* pixelPtr, const Color& color, int samplesPerPixel) {
    float scale = 1.0f / samplesPerPixel;

    pixelPtr[0] = (uchar8) (256 * clamp(color[0] * scale, 0.0f, 0.999f));
    pixelPtr[1] = (uchar8) (256 * clamp(color[1] * scale, 0.0f, 0.999f));
    pixelPtr[2] = (uchar8) (256 * clamp(color[2] * scale, 0.0f, 0.999f));
}

static __device__ Color backgroundAttenuation(const Ray& ray) {
    Vector3D unitDir = ray.direction().normalized();
    float t = 0.5f * (unitDir.y() + 1.0f);
    return (1.0f - t) * Color(1.0f, 1.0f, 1.0f) + t * Color(0.5f, 0.7f, 1.0f);
}

__device__ Color getColor(const Ray& ray, Hittable* const* world,
                          int maxDepth, curandState* randState) {
    Color color(1.0f, 1.0f, 1.0f);
    Ray currRay = ray;

    for (int i = 0; i < maxDepth; i++) {
        HitRecord record;

        if (!(*world)->hit(currRay, 0.001f, GlobalConstants::infinity, record)) {
            return backgroundAttenuation(currRay) * color;
        }

        Ray scattered;
        Color attenuation;
        if (record.material->scatter(currRay, record, attenuation, scattered, randState)) {
            color *= attenuation;
            currRay = scattered;

        } else {
            return Color();
        }
    }

    return Color();
}

__global__ void pixelRender(const RenderInfo ri, uchar8* colorData,
                            curandState* randStateArr,
                            const Camera* cam, Hittable* const* world) {

    uint32 x = threadIdx.x + blockIdx.x * blockDim.x;
    uint32 y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= ri.imgWidth || y >= ri.imgHeight) { return; }

    uint32 pixelIdx = ri.imgWidth * y + x;
    curandState localRandState = randStateArr[pixelIdx];

    uint32 yMapped = ri.imgHeight - y - 1;

    Color pixelColor;
    for (int i = 0; i < ri.samplesPerPixel; i++) {
        float xDisturbed = x + randomFloat(&localRandState);
        float yDisturbed = yMapped + randomFloat(&localRandState);

        float u = xDisturbed / (float) (ri.imgWidth - 1);
        float v = yDisturbed / (float) (ri.imgHeight - 1);

        Ray ray = cam->getRay(u, v);
        pixelColor += getColor(ray, world, ri.maxDepth, &localRandState);
    }

    writeColor(&colorData[3 * pixelIdx], pixelColor, ri.samplesPerPixel);
}

__global__ void initRandomState(int imgWidth, int imgHeight, uint32 firstSeed,
                           curandState* randStateArr) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if((x >= imgWidth) || (y >= imgHeight)) return;

    uint32 pixelIdx = y * imgWidth + x;
    curand_init(firstSeed + pixelIdx, 0, 0, &randStateArr[pixelIdx]);
}

__global__ void createWorld(Hittable** world) {
    Hittable** list = new Hittable*[2];

    auto* material = new Lambertian(Color(0.5f, 0.5f, 0.5f));

    list[0] = new Sphere(Point3D(0.0f, 0.0f, -1.0f), 0.5f, material);
    list[1] = new Sphere(Point3D(0.0f, -100.5f, -1.0f), 100.0f, material);

    *world = new HittableList(list, 2);
}

__global__ void destroyWorld(Hittable** world) {
    delete *world;
}

Renderer::Renderer(const RenderInfo& renderInfo) : mRi(renderInfo) {
    int imgSquare = mRi.imgWidth * mRi.imgHeight;

    mColorDataSize = 3 * imgSquare;
    catchErrorInClass(cudaMalloc(&mColorData_d, mColorDataSize * sizeof(uchar8)));
    mColorData_h = new uchar8[mColorDataSize];

    catchErrorInClass(cudaMalloc(&mWorld_d, sizeof(Hittable*)));
    createWorld<<<1, 1>>>(mWorld_d);
    checkErrorInClass("createWorld");

    catchErrorInClass(cudaMalloc(&mRandStateArr, imgSquare * sizeof(curandState)));

    uint32 seed = (uint32)time(nullptr);

    int gridWidth = (mRi.imgWidth + threadBlockWidth - 1) / threadBlockWidth;
    int gridHeight = (mRi.imgHeight + threadBlockHeight - 1) / threadBlockHeight;

    mGridDim = dim3(gridWidth, gridHeight);
    mBlockDim = dim3(threadBlockWidth, threadBlockHeight);

    initRandomState<<<mGridDim, mBlockDim>>>(mRi.imgWidth, mRi.imgHeight, seed, mRandStateArr);
    checkErrorInClass("initRandomState");
}

uchar8* Renderer::renderRaw(const Camera* camera) {
    clock_t start, stop;
    start = clock();

    pixelRender<<<mGridDim, mBlockDim>>>(mRi, mColorData_d, mRandStateArr, camera, mWorld_d);
    checkError("pixelRender");

    catchError(cudaMemcpy(mColorData_h, mColorData_d,
                                 mColorDataSize * sizeof(uchar8), cudaMemcpyDeviceToHost));

    stop = clock();
    double timerSeconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cout << "took " << timerSeconds << " seconds.\n";

    return mColorData_h;
}

Renderer::~Renderer() {
    destroyWorld<<<1, 1>>>(mWorld_d);
    checkError("destroyWorld");

    catchError(cudaFree(mWorld_d));
    catchError(cudaFree(mColorData_d));
    catchError(cudaFree(mRandStateArr));

    delete mColorData_h;
}