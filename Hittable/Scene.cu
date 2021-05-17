#include "Scene.h"
#include "HittableList.h"

__global__ void createOnGPU(Hittable** scene,
                            HittableDef** hitDefs,
                            MaterialDef** matDefs,
                            size_t size) {
    Hittable** list = new Hittable*[size];

    for (size_t i = 0; i < size; i++) {
        auto* material = createMaterial(matDefs[i]);
        list[i] = createHittable(hitDefs[i], material);
    }

    *scene = new HittableList(list, size);
}


Hittable** Scene::create(HittableDef** hitDefs,
                         MaterialDef** matDefs,
                         size_t size) {
    Hittable** scene;
    catchError(cudaMalloc(&scene, sizeof(Hittable*)));
    createOnGPU<<<1, 1>>>(scene, hitDefs, matDefs, size);
    return scene;
}
