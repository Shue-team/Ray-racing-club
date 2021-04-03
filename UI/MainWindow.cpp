//
// Created by arseny on 06.02.2021.
//

// You may need to build the project (run Qt uic code generator) to get "ui_MainWindow.h" resolved

#include <QAction>
#include <QDebug>

#include <iostream>

#include "MainWindow.h"
#include "ui_MainWindow.h"

constexpr int imgWidth = 700;
constexpr int imgHeight = 450;
constexpr int samplesPerPixel = 10;
constexpr int maxDepth = 50;

MainWindow::MainWindow(QWidget* parent) :
        QWidget(parent), mUi(new Ui::MainWindow) {
    mUi->setupUi(this);

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

void MainWindow::onRenderAction() const {
    if (!mRenderer->isValid()) { return; }

    float aspectRatio = imgWidth / (float) imgHeight;
    auto* camera = new Camera(aspectRatio);

    QImage image = mRenderer->render(camera);
    mUi->imageDisplay->setPixmap(QPixmap::fromImage(image));
}
