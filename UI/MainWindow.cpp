//
// Created by arseny on 06.02.2021.
//

// You may need to build the project (run Qt uic code generator) to get "ui_MainWindow.h" resolved

#include <QAction>
#include <QDebug>

#include <iostream>

#include <QMouseEvent>
#include "MainWindow.h"
#include "ui_MainWindow.h"
#include "../Render/Camera.h"

constexpr int imgWidth = 700;
constexpr int imgHeight = 450;
constexpr int samplesPerPixel = 10;
constexpr int maxDepth = 50;

constexpr float moveStep = 0.1f;
constexpr float rotStep = 0.05f;
constexpr float zoomStep = 0.05f;
constexpr float mouseAcc = 0.001f;


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

    CamParams params;
    params.lookFrom = { 0, 0, 0 };
    params.lookAt = { 0, 0, -1 };
    params.vUp = { 0, 1, 0 };
    params.aspectRatio = (float)imgWidth / imgHeight;
    params.vfov = 90;
    params.aperture = 0.1;
    params.focusDist = 1;
    mCamera = new Camera(params);

    Controller::Steps steps = { moveStep, rotStep, zoomStep, mouseAcc };
    mController = new Controller(mCamera, steps, this);
    installEventFilter(mController);

    auto renderAction = new QAction(this);
    renderAction->setShortcut(Qt::Key_F);
    connect(renderAction, &QAction::triggered, this, &MainWindow::toggleMouseTracking);
    addAction(renderAction);
    
    renderAction = new QAction(this);
    renderAction->setShortcut(Qt::CTRL + Qt::Key_S);
    connect(renderAction, &QAction::triggered, this, &MainWindow::querySave);
    addAction(renderAction);
}

MainWindow::~MainWindow() {
    delete mUi;
    delete mRenderer;
    delete mCamera;
}

void MainWindow::paintEvent(QPaintEvent *event) {
    if (!mRenderer->isValid()) { return; }
    float timeTook = 0;
    QImage image = mRenderer->render(mCamera, timeTook);
    mUi->imageDisplay->setPixmap(QPixmap::fromImage(image));
    this->setWindowTitle(QString("FPS: ") + QString::number(1.f / timeTook, 'f', 2));
    if (fSave) {
        if (!image.save("Screenshot.png", "PNG")) {
            std::cout << "Error during screenshot save" << std::endl;
        }
        fSave = false;
    }
}

void MainWindow::toggleMouseTracking() {
    fTracking = !fTracking;
    setMouseTracking(fTracking);
    mUi->imageDisplay->setMouseTracking(fTracking);
    if (fTracking) {
        grabMouse(); // works not as we thought
    }
    else {
        releaseMouse();
    }
}

void MainWindow::querySave() {
    fSave = true;
}
