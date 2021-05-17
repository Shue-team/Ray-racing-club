#ifndef WORLDEDIT_H
#define WORLDEDIT_H

#include <QComboBox>
#include <QListWidget>
#include <QMap>
#include <QPushButton>
#include <QVBoxLayout>
#include <QWidget>

#include "HittableEdit.h"
#include "MaterialEdit.h"

class WorldEdit : public QWidget
{
    Q_OBJECT
public:
    explicit WorldEdit(QWidget *parent = nullptr);

    Hittable** createWorld() const;

private slots:
    void onHittableChanged(const QString& name);
    void onMaterialChanged(const QString& name);

    void onHittableDataFilled(bool flag);
    void onMaterialDataFilled(bool flag);

    void onAddButton();

private:
    QComboBox* initHittableMap();
    QComboBox* initMaterialMap();

    bool mHittableDataFilled = false;
    bool mMaterialDataFilled = false;

    QListWidget* mObjectsList;
    QVBoxLayout* mLayout;

    QMap<QString, HittableEdit*> mHittableMap;
    QMap<QString, MaterialEdit*> mMaterialMap;

    QPushButton* mAddButton;

    QVector<HittableDef*> mHittableDefs;
    QVector<MaterialDef*> mMaterialDefs;
};

#endif // WORLDEDIT_H
