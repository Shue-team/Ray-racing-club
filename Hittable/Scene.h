#ifndef SCENE_H
#define SCENE_H

#include "HittableDef.h"
#include "../Material/MaterialDef.h"

namespace Scene {
    Hittable** create(HittableDef** hitDefs,
                      MaterialDef** matDefs,
                      size_t size);
}

#endif // SCENE_H
