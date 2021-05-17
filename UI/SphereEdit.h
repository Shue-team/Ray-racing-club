#ifndef SPHEREEDIT_H
#define SPHEREEDIT_H

#include "HittableEdit.h"

#include <QLineEdit>
#include <QVBoxLayout>

class SphereEdit : public HittableEdit
{
public:
    explicit SphereEdit(QWidget *parent = nullptr);

    HittableDef* createDefinition() const override;

private slots:
    void onTextEdited();

private:
    void initCenterField(QGridLayout* layout);
    void initRadiusField(QGridLayout* layout);

    QList<QLineEdit*> mCoordsEdits;
    QLineEdit* mRadiusEdit;
};

#endif // SPHEREEDIT_H
