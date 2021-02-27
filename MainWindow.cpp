//
// Created by arseny on 06.02.2021.
//

// You may need to build the project (run Qt uic code generator) to get "ui_MainWindow.h" resolved

#include <QAction>
#include <QDebug>

#include "MainWindow.h"
#include "ui_MainWindow.h"

constexpr int imgWidth = 800;
constexpr int imgHeight = 600;

MainWindow::MainWindow(QWidget* parent) :
        QWidget(parent), mUi(new Ui::MainWindow) {
    mUi->setupUi(this);

    mRenderer = new Renderer(imgWidth, imgHeight, 10);

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
    float aspectRatio = imgWidth / imgHeight;
    auto* camera = new Camera(aspectRatio);

    uchar8* pixelData = mRenderer->render(camera);

    QImage image(pixelData, imgWidth, imgHeight, imgWidth * 3, QImage::Format_RGB888);
    mUi->imageDisplay->setPixmap(QPixmap::fromImage(image));
}
