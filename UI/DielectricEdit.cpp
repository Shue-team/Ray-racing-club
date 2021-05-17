#include "DielectricEdit.h"
#include "../Material/Dielectric.h"

#include <QDoubleValidator>
#include <QLabel>

DielectricEdit::DielectricEdit(QWidget *parent)
    : MaterialEdit(parent) {
    auto* layout = new QGridLayout(this);

    auto* refractIdxLabel = new QLabel("Коэффициент приломления:");

    layout->addWidget(refractIdxLabel, 0, 0, 1, 2);

    mRefractIdxEdit = new QLineEdit;

    auto* validator = new QDoubleValidator(mRefractIdxEdit);
    validator->setLocale(QLocale::English);
    mRefractIdxEdit->setValidator(validator);

    connect(mRefractIdxEdit, &QLineEdit::textEdited,
            this, &DielectricEdit::onTextEdited);

    layout->addWidget(mRefractIdxEdit, 1, 0);
    layout->setColumnStretch(1, 2);
    layout->setColumnStretch(0, 1);
}

MaterialDef* DielectricEdit::createDefinition() const {
    double refractIdx = mRefractIdxEdit->text().toDouble();

    auto* hostDef = new DielectricDef(refractIdx);
    auto* deviceDef = MaterialDef::transferToGPU(hostDef);
    delete hostDef;

    return deviceDef;
}

void DielectricEdit::onTextEdited() {
    bool isDataFilled = !mRefractIdxEdit->text().isEmpty();
    emit dataFilled(isDataFilled);
}
