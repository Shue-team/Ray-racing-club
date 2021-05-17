#include <QLabel>
#include <QPushButton>
#include <QTextEdit>

#include "DielectricEdit.h"
#include "LambertianEdit.h"
#include "MetalEdit.h"
#include "SphereEdit.h"
#include "TriangleEdit.h"
#include "WorldEdit.h"
#include "../ErrorProcessing/ErrorHandling.h"
#include "../Hittable/Scene.h"

constexpr int itemStretch = 8;
constexpr int hittableItemIdx = 4;
constexpr int materialItemIdx = 7;

WorldEdit::WorldEdit(QWidget *parent)
    : QWidget(parent) {
    initHittableMap();

    mLayout = new QVBoxLayout(this);
    mLayout->setContentsMargins(0, 0, 0, 0);

    auto* listHeader = new QLabel("Список примитивов");
    listHeader->setAlignment(Qt::AlignCenter);
    mLayout->addWidget(listHeader, 1);

    mObjectsList = new QListWidget;
    mLayout->addWidget(mObjectsList, itemStretch * 2);

    auto* hittableBoxHeader = new QLabel("Выберите примитив...");
    hittableBoxHeader->setAlignment(Qt::AlignCenter);
    mLayout->addWidget(hittableBoxHeader, 1);

    mLayout->addWidget(initHittableMap());

    mLayout->insertWidget(hittableItemIdx, mHittableMap.first(),
                          itemStretch, Qt::AlignTop);

    auto* materialBoxHeader = new QLabel("Выберите материал...");
    materialBoxHeader->setAlignment(Qt::AlignCenter);
    mLayout->addWidget(materialBoxHeader, 1);

    mLayout->addWidget(initMaterialMap());

    mLayout->insertWidget(materialItemIdx, mMaterialMap.first(),
                          itemStretch, Qt::AlignTop);

    QPalette pal(palette());
    pal.setColor(QPalette::Window, Qt::white);
    setAutoFillBackground(true);
    setPalette(pal);

    mAddButton = new QPushButton("Добавить");
    mAddButton->setEnabled(false);
    mLayout->addWidget(mAddButton, Qt::AlignCenter);
    mLayout->addStretch(1);

    connect(mAddButton, &QPushButton::clicked,
            this, &WorldEdit::onAddButton);
}

Hittable** WorldEdit::createWorld() const {
    size_t n = mHittableDefs.size();

    HittableDef** hittableDefs;
    catchError(cudaMalloc(&hittableDefs, n * sizeof(HittableDef*)));
    catchError(cudaMemcpy(hittableDefs, mHittableDefs.data(),
                          n * sizeof(HittableDef*), cudaMemcpyHostToDevice));

    MaterialDef** materialDefs;
    catchError(cudaMalloc(&materialDefs, n * sizeof(MaterialDef*)));
    catchError(cudaMemcpy(materialDefs, mMaterialDefs.data(),
                          n * sizeof(MaterialDef*), cudaMemcpyHostToDevice));

    auto* scene = Scene::create(hittableDefs, materialDefs, n);
    checkError("Scene::create");
    return scene;
}

void WorldEdit::onHittableChanged(const QString& name)
{
    auto* item = mLayout->itemAt(hittableItemIdx);
    if (auto* widget = item->widget()) {
        mLayout->replaceWidget(widget, mHittableMap[name]);
        widget->setParent(nullptr);
    }
}

void WorldEdit::onMaterialChanged(const QString &name)
{
    auto* item = mLayout->itemAt(materialItemIdx);
    if (auto* widget = item->widget()) {
        mLayout->replaceWidget(widget, mMaterialMap[name]);
        widget->setParent(nullptr);
    }
}

void WorldEdit::onHittableDataFilled(bool flag){
    mHittableDataFilled = flag;

    bool addReady = mHittableDataFilled && mMaterialDataFilled;
    mAddButton->setEnabled(addReady);
}

void WorldEdit::onMaterialDataFilled(bool flag) {
    mMaterialDataFilled = flag;

    bool addReady = mHittableDataFilled && mMaterialDataFilled;
    mAddButton->setEnabled(addReady);
}

void WorldEdit::onAddButton() {
    auto* hittableItem = mLayout->itemAt(hittableItemIdx);
    auto* hittableEdit = static_cast<HittableEdit*>(hittableItem->widget());
    auto* hittableDef = hittableEdit->createDefinition();

    auto* materialItem = mLayout->itemAt(materialItemIdx);
    auto* materialEdit = static_cast<MaterialEdit*>(materialItem->widget());
    auto* materialDef = materialEdit->createDefinition();

    mMaterialDefs.append(materialDef);
    mHittableDefs.append(hittableDef);

    mObjectsList->addItem(QString("Object%1")
                          .arg(mObjectsList->count() + 1));
}

QComboBox *WorldEdit::initHittableMap()
{
    mHittableMap["Сфера"] = new SphereEdit;
    mHittableMap["Треугольник"] = new TriangleEdit;

    for (auto* edit : mHittableMap) {
        connect(edit, &HittableEdit::dataFilled,
                this, &WorldEdit::onHittableDataFilled);
    }

    auto* objectComboBox = new QComboBox;

    for (auto& name : mHittableMap.keys()) {
        objectComboBox->addItem(name);
    }

    connect(objectComboBox, &QComboBox::currentTextChanged,
            this, &WorldEdit::onHittableChanged);

    return objectComboBox;
}

QComboBox *WorldEdit::initMaterialMap(){
    mMaterialMap["Метал"] = new MetalEdit;
    mMaterialMap["Диэлектрик"] = new DielectricEdit;
    mMaterialMap["Диффузная поверхность"] = new LambertianEdit;

    for (auto* edit : mMaterialMap) {
        connect(edit, &MaterialEdit::dataFilled,
                this, &WorldEdit::onMaterialDataFilled);
    }

    auto* materialComboBox = new QComboBox;

    for (auto& name : mMaterialMap.keys()) {
        materialComboBox->addItem(name);
    }

    connect(materialComboBox, &QComboBox::currentTextChanged,
            this, &WorldEdit::onMaterialChanged);

    return materialComboBox;
}

