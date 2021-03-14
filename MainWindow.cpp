//
// Created by arseny on 06.02.2021.
//

// You may need to build the project (run Qt uic code generator) to get "ui_MainWindow.h" resolved

#include <QAction>
#include <QDebug>

#include <iostream>

#include "MainWindow.h"
#include "ui_MainWindow.h"

constexpr int imgWidth = 1280;
constexpr int imgHeight = 720;
constexpr int samplesPerPixel = 10;

/*
 * wasd - moving in Oxy plane
 * arrows - rotating camera
 * e - zoom in
 * q - zoom out
*/
MainWindow::MainWindow(QWidget* parent) :
        QWidget(parent), mUi(new Ui::MainWindow) {
    mUi->setupUi(this);

    RenderInfo renderInfo;
    renderInfo.imgWidth = imgWidth;
    renderInfo.imgHeight = imgHeight;
    renderInfo.samplesPerPixel = samplesPerPixel;

    mRenderer = new Renderer(renderInfo);
    if (!mRenderer->isValid()) {
        std::cerr << "Renderer wasn't created correctly" << std::endl;
    }
    mCamera = new Camera((float) imgWidth / imgHeight);

    // render is automatic to test the camera
    /*
    auto renderAction = new QAction(this);
    renderAction->setShortcut(Qt::Key_R);
    connect(renderAction, &QAction::triggered, this, &MainWindow::onRenderAction);
    addAction(renderAction);
    */
    auto renderAction = new QAction(this);
    renderAction->setShortcut(Qt::Key_W);
    connect(renderAction, &QAction::triggered, this, &MainWindow::moveUp);
    addAction(renderAction);

    renderAction = new QAction(this);
    renderAction->setShortcut(Qt::Key_S);
    connect(renderAction, &QAction::triggered, this, &MainWindow::moveDown);
    addAction(renderAction);

    renderAction = new QAction(this);
    renderAction->setShortcut(Qt::Key_A);
    connect(renderAction, &QAction::triggered, this, &MainWindow::moveLeft);
    addAction(renderAction);

    renderAction = new QAction(this);
    renderAction->setShortcut(Qt::Key_D);
    connect(renderAction, &QAction::triggered, this, &MainWindow::moveRight);
    addAction(renderAction);

    renderAction = new QAction(this);
    renderAction->setShortcut(Qt::Key_Up);
    connect(renderAction, &QAction::triggered, this, &MainWindow::rotateUp);
    addAction(renderAction);

    renderAction = new QAction(this);
    renderAction->setShortcut(Qt::Key_Down);
    connect(renderAction, &QAction::triggered, this, &MainWindow::rotateDown);
    addAction(renderAction);

    renderAction = new QAction(this);
    renderAction->setShortcut(Qt::Key_Left);
    connect(renderAction, &QAction::triggered, this, &MainWindow::rotateLeft);
    addAction(renderAction);

    renderAction = new QAction(this);
    renderAction->setShortcut(Qt::Key_Right);
    connect(renderAction, &QAction::triggered, this, &MainWindow::rotateRight);
    addAction(renderAction);

    renderAction = new QAction(this);
    renderAction->setShortcut(Qt::Key_E);
    connect(renderAction, &QAction::triggered, this, &MainWindow::zoomIn);
    addAction(renderAction);

    renderAction = new QAction(this);
    renderAction->setShortcut(Qt::Key_Q);
    connect(renderAction, &QAction::triggered, this, &MainWindow::zoomOut);
    addAction(renderAction);

}

MainWindow::~MainWindow() {
    delete mUi;
    delete mRenderer;
    delete mCamera;
}

void MainWindow::paintEvent(QPaintEvent *event) {
    if (!mRenderer->isValid()) { return; }
    QImage image = mRenderer->render(mCamera);
    mUi->imageDisplay->setPixmap(QPixmap::fromImage(image));
}

// render is automatic to test the camera
/*
void MainWindow::onRenderAction() {
    if (!mRenderer->isValid()) { return; }
    update();
}
*/

void MainWindow::moveLeft() {
    mCamera->moveHorz(-0.1f);
    update();
}

void MainWindow::moveRight() {
    mCamera->moveHorz(0.1f);
    update();
}

void MainWindow::moveDown() {
    mCamera->moveVert(-0.1f);
    update();
}

void MainWindow::moveUp() {
    mCamera->moveVert(0.1f);
    update();
}

void MainWindow::rotateUp() {
    mCamera->rotateOx(0.1f);
    update();
}

void MainWindow::rotateDown() {
    mCamera->rotateOx(-0.1f);
    update();
}

void MainWindow::rotateLeft() {
    mCamera->rotateOy(0.1f);
    update();
}

void MainWindow::rotateRight() {
    mCamera->rotateOy(-0.1f);
    update();
}

void MainWindow::zoomIn() {
    mCamera->moveDepth(-0.1f);
    update();
}

void MainWindow::zoomOut() {
    mCamera->moveDepth(0.1f);
    update();
}
