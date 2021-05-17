#ifndef TRIANGLEEDIT_H
#define TRIANGLEEDIT_H

#include "HittableEdit.h"

class TriangleEdit : public HittableEdit
{
public:
    TriangleEdit(QWidget* parent = nullptr);

    HittableDef* createDefinition() const override {}
};

#endif // TRIANGLEEDIT_H
