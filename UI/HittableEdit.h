#ifndef HITTABLEEDIT_H
#define HITTABLEEDIT_H

#include <QWidget>

#include "../Hittable/HittableDef.h"

class HittableEdit : public QWidget
{
    Q_OBJECT
public:
    explicit HittableEdit(QWidget *parent = nullptr)
        : QWidget(parent) {}

    virtual HittableDef* createDefinition() const = 0;

signals:
    void dataFilled(bool flag);
};

#endif // HITTABLEEDIT_H
