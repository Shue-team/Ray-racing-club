#include "TriangleEdit.h"

TriangleEdit::TriangleEdit(QWidget* parent)
    : HittableEdit(parent) {}

HittableDef* TriangleEdit::createDefinition() const {
    return nullptr;
}
