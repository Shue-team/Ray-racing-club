#include "MetalEdit.h"
#include "../Material/Metal.h"

#include <QDoubleValidator>
#include <QLabel>

MetalEdit::MetalEdit(QWidget* parent) : MaterialEdit(parent) {
    auto* layout = new QGridLayout(this);

    initAlbedoField(layout);
    initFuzzField(layout);
}

MaterialDef *MetalEdit::createDefinition() const {
    double r = mAlbedoEdits[0]->text().toDouble();
    double g = mAlbedoEdits[1]->text().toDouble();
    double b = mAlbedoEdits[2]->text().toDouble();

    double fuzz = mFuzzEdit->text().toDouble();

    auto* hostDef = new MetalDef(Color(r, g, b), fuzz);
    auto* deviceDef = MaterialDef::transferToGPU(hostDef);
    delete hostDef;

    return deviceDef;
}

void MetalEdit::onTextEdited() {
    bool isDataFilled = true;
    for (auto* edit : mAlbedoEdits) {
        isDataFilled &= !edit->text().isEmpty();
    }

    isDataFilled &= !mFuzzEdit->text().isEmpty();

    emit dataFilled(isDataFilled);
}

void MetalEdit::initAlbedoField(QGridLayout *layout) {
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
                this, &MetalEdit::onTextEdited);
    }
}

void MetalEdit::initFuzzField(QGridLayout *layout)
{
    auto* fuzzLabel = new QLabel("Рассеивание:");

    layout->addWidget(fuzzLabel, 2, 0);

    mFuzzEdit = new QLineEdit;

    auto* validator = new QDoubleValidator(mFuzzEdit);
    validator->setLocale(QLocale::English);
    mFuzzEdit->setValidator(validator);

    layout->addWidget(mFuzzEdit, 3, 0);

    connect(mFuzzEdit, &QLineEdit::textEdited,
            this, &MetalEdit::onTextEdited);
}
