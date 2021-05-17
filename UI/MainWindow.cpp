//
// Created by arseny on 06.02.2021.
//

// You may need to build the project (run Qt uic code generator) to get "ui_MainWindow.h" resolved

#include <QAction>
#include <QGraphicsPixmapItem>
#include <QPushButton>

#include <iostream>

#include "MainWindow.h"
#include "WorldEdit.h"
#include "ui_MainWindow.h"

constexpr int imgWidth = 700;
constexpr int imgHeight = 450;
constexpr int samplesPerPixel = 10;
constexpr int maxDepth = 50;

MainWindow::MainWindow(QWidget* parent) :
        QWidget(parent), mUi(new Ui::MainWindow) {
    mUi->setupUi(this);

    initWorldEdit();

    RenderInfo renderInfo;
    renderInfo.imgWidth = imgWidth;
    renderInfo.imgHeight = imgHeight;
    renderInfo.samplesPerPixel = samplesPerPixel;
    renderInfo.maxDepth = maxDepth;

    mRenderer = new Renderer(renderInfo);
    if (!mRenderer->isValid()) {
        std::cerr << "Renderer wasn't created correctly" << std::endl;
    }

    auto renderAction = new QAction(this);
    renderAction->setShortcut(Qt::Key_R);
    connect(renderAction, &QAction::triggered, this, &MainWindow::onRenderAction);
    addAction(renderAction);
}

MainWindow::~MainWindow() {
    delete mUi;
    delete mRenderer;
}

void MainWindow::initWorldEdit()
{
    mWorldEdit = new WorldEdit;

    auto* grid = static_cast<QGridLayout*>(layout());

    auto* maskLayout = new QHBoxLayout;

    auto* worldEditButton = new QPushButton;

    worldEditButton->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred);
    worldEditButton->setFixedWidth(20);

    maskLayout->addWidget(mWorldEdit, 2);
    maskLayout->addWidget(worldEditButton);
    maskLayout->addStretch(5);

    grid->addLayout(maskLayout, 0, 0);

    connect(worldEditButton, &QPushButton::clicked, this, &MainWindow::onWorldEditButton);
}

void MainWindow::onWorldEditButton()
{
    bool isVisible = mWorldEdit->isVisible();
    mWorldEdit->setVisible(!isVisible);
}

void MainWindow::onRenderAction() const {
    if (!mRenderer->isValid()) { return; }

    float aspectRatio = imgWidth / (float) imgHeight;
    auto* camera = new Camera(aspectRatio);

    mRenderer->setWorld(mWorldEdit->createWorld());

    uchar* imgData = mRenderer->render(camera);
    QImage image(imgData, imgWidth, imgHeight, 3 * imgWidth, QImage::Format_RGB888);

    mUi->imageDisplay->setPixmap(QPixmap::fromImage(image));
}
