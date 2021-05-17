#include "LambertianEdit.h"
#include "../Material/Lambertian.h"

#include <QDoubleValidator>
#include <QLabel>

LambertianEdit::LambertianEdit(QWidget* parent)
    : MaterialEdit(parent) {
    auto* layout = new QGridLayout(this);

    auto* albedoLabel = new QLabel("Альбедо:");

    layout->addWidget(albedoLabel, 0, 0);

    for (int i = 0; i < 3; i++) {
        auto* albedoEdit = new QLineEdit;

        auto* validator = new QDoubleValidator(albedoEdit);
        validator->setLocale(QLocale::English);
        albedoEdit->setValidator(validator);

        layout->addWidget(albedoEdit, 1, i);
        mAlbedoEdits.append(albedoEdit);

        connect(albedoEdit, &QLineEdit::textEdited,
                this, &LambertianEdit::onTextEdited);
    }
}

MaterialDef *LambertianEdit::createDefinition() const {
    double r = mAlbedoEdits[0]->text().toDouble();
    double g = mAlbedoEdits[1]->text().toDouble();
    double b = mAlbedoEdits[2]->text().toDouble();

    auto* hostDef = new LambertianDef(Point3D(r, g, b));
    auto* deviceDef = MaterialDef::transferToGPU(hostDef);
    delete hostDef;

    return deviceDef;
}

void LambertianEdit::onTextEdited() {
    bool isDataFilled = true;
    for (auto* edit : mAlbedoEdits) {
        isDataFilled &= !edit->text().isEmpty();
    }

    emit dataFilled(isDataFilled);
}
