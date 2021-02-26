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

    auto renderAction = new QAction(this);
    renderAction->setShortcut(Qt::Key_R);
    connect(renderAction, &QAction::triggered, this, &MainWindow::onRenderAction);
    addAction(renderAction);
}

MainWindow::~MainWindow() {
    delete mUi;
}

void MainWindow::onRenderAction() const {
    int width = mUi->imageDisplay->width();
    int height = mUi->imageDisplay->height();

    Renderer renderer(width, height, 100);

    float aspectRatio = width / height;
    auto* camera = new Camera(aspectRatio);

    uchar8* pixelData = renderer.render(camera);

    QImage image(pixelData, width, height,QImage::Format_RGB888);
    mUi->imageDisplay->setPixmap(QPixmap::fromImage(image));
}
