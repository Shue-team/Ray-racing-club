#include "Vector3D.h"
#include "Hittable/Sphere.h"

//TODO (1): Написать __global__ функцию для запуска GPU, которая бы производила рендер изображения
// In: Hittable* - указатель на "мир" объектов
// In: Camera* - экземпляр класса камеры для создания лучей
// In, Out: Color* - Массив, заполенный нулевыми значениями, для хранения цветов пикселей
// В одном блоке потоков создаются все лучи, отвечающие за рендер одного пикселя,
// результат каждого суммируется в соотвуствующую ячейку массива при помощи Vector3D::atomicAdd().
// Все входные параметры обязательно(!) должны быть алоцированны на памяти видеокарты, иначе разыменование данных указателей будет невозможно.