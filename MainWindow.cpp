//
// Created by arseny on 06.02.2021.
//

// You may need to build the project (run Qt uic code generator) to get "ui_MainWindow.h" resolved

#include <QAction>
#include <QDebug>

#include "MainWindow.h"
#include "ui_MainWindow.h"

#include "Renderer.h"
#include <stdio.h>


MainWindow::MainWindow(QWidget* parent) :
        QWidget(parent), mUi(new Ui::MainWindow) {
    mUi->setupUi(this);

    /*mRenderer = new Renderer(mUi->imageDisplay->height(), mUi->imageDisplay->width());
    if (!mRenderer) {
        qDebug() << "not enough memory for renderer!";
    }*/

    auto renderAction = new QAction(this);
    renderAction->setShortcut(Qt::Key_R);
    connect(renderAction, &QAction::triggered, this, &MainWindow::onRenderAction);
    addAction(renderAction);
    auto modeAction = new QAction(this);
    modeAction->setShortcut(Qt::Key_E);
    connect(modeAction, &QAction::triggered, this, &MainWindow::onModeChangeAction);
    addAction(modeAction);
}

MainWindow::~MainWindow() {
    //delete mRenderer;
    cudaError_t cudaStatus = cudaDeviceReset();
    qDebug() << "device reset status: " << cudaGetErrorString(cudaStatus);
    delete mUi;
}

void MainWindow::onRenderAction() const {
    Renderer renderer(mUi->imageDisplay->height(), mUi->imageDisplay->width()); // temporarily
    Renderer* mRenderer = &renderer;
    int height = mRenderer->getHeight();
    int width = mRenderer->getWidth();
    if (fChangeMode) { // temporarily
        mRenderer->changeMode();
    }
    unsigned char* pixels = mRenderer->render();
    if (!pixels) {
        qDebug() << "render returned nullptr";
        return;
    }
    QImage image(pixels, width, height, 3 * width, mRenderer->getImageFormat());
    mUi->imageDisplay->setPixmap(QPixmap::fromImage(image));
}

void MainWindow::onModeChangeAction() {
    fChangeMode = !fChangeMode;
}
