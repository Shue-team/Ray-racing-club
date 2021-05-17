#ifndef DIELECTRICEDIT_H
#define DIELECTRICEDIT_H

#include "MaterialEdit.h"

#include <QGridLayout>
#include <QLineEdit>

class DielectricEdit : public MaterialEdit {

public:
    DielectricEdit(QWidget* parent = nullptr);

    MaterialDef* createDefinition() const override;

private slots:
    void onTextEdited();

private:
    QLineEdit* mRefractIdxEdit;
};

#endif // DIELECTRICEDIT_H
