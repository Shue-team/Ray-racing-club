#ifndef METALEDIT_H
#define METALEDIT_H

#include "MaterialEdit.h"

#include <QGridLayout>
#include <QLineEdit>

class MetalEdit : public MaterialEdit {

public:
    MetalEdit(QWidget* parent = nullptr);

    MaterialDef* createDefinition() const override;

private slots:
    void onTextEdited();

private:
    void initAlbedoField(QGridLayout* layout);
    void initFuzzField(QGridLayout* layout);

    QList<QLineEdit*> mAlbedoEdits;
    QLineEdit* mFuzzEdit;
};

#endif // METALEDIT_H
