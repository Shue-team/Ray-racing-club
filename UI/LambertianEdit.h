#ifndef LAMBERTIANEDIT_H
#define LAMBERTIANEDIT_H

#include "MaterialEdit.h"

#include <QGridLayout>
#include <QLineEdit>

class LambertianEdit : public MaterialEdit {

public:
    LambertianEdit(QWidget* parent = nullptr);

    MaterialDef* createDefinition() const override;

private slots:
    void onTextEdited();

private:
    QList<QLineEdit*> mAlbedoEdits;
};


#endif // LAMBERTIANEDIT_H
