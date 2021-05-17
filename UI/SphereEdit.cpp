#include "SphereEdit.h"
#include "../Hittable/Sphere.h"

#include <QDoubleValidator>
#include <QLabel>
#include <QVBoxLayout>

SphereEdit::SphereEdit(QWidget *parent) : HittableEdit(parent) {
    auto* layout = new QGridLayout(this);

    initCenterField(layout);
    initRadiusField(layout);
}

HittableDef *SphereEdit::createDefinition() const {
    double x = mCoordsEdits[0]->text().toDouble();
    double y = mCoordsEdits[1]->text().toDouble();
    double z = mCoordsEdits[2]->text().toDouble();

    double radius = mRadiusEdit->text().toDouble();

    auto* hostDef = new SphereDef(Point3D(x, y, z), radius);
    auto* deviceDef = HittableDef::transferToGPU(hostDef);
    delete hostDef;

    return deviceDef;
}

void SphereEdit::onTextEdited()
{
    bool isDataFilled = true;
    for (auto* edit : mCoordsEdits) {
        isDataFilled &= !edit->text().isEmpty();
    }

    isDataFilled &= !mRadiusEdit->text().isEmpty();

    emit dataFilled(isDataFilled);
}

void SphereEdit::initCenterField(QGridLayout* layout)
{
    auto* centerLabel = new QLabel("Центр:");

    layout->addWidget(centerLabel, 0, 0);

    QStringList axisNames = {"x", "y", "z"};
    for (int i = 0; i < axisNames.size(); i++) {
        auto* coordEdit = new QLineEdit;
        coordEdit->setPlaceholderText(axisNames[i]);

        auto* validator = new QDoubleValidator(coordEdit);
        validator->setLocale(QLocale::English);
        coordEdit->setValidator(validator);

        layout->addWidget(coordEdit, 1, i);
        mCoordsEdits.append(coordEdit);

        connect(coordEdit, &QLineEdit::textEdited,
                this, &SphereEdit::onTextEdited);
    }
}

void SphereEdit::initRadiusField(QGridLayout* layout) {
    auto* radiusLabel = new QLabel("Радиус:");

    layout->addWidget(radiusLabel, 2, 0);

    mRadiusEdit = new QLineEdit;

    auto* validator = new QDoubleValidator(mRadiusEdit);
    validator->setLocale(QLocale::English);
    mRadiusEdit->setValidator(validator);

    layout->addWidget(mRadiusEdit, 3, 0);

    connect(mRadiusEdit, &QLineEdit::textEdited,
            this, &SphereEdit::onTextEdited);
}
