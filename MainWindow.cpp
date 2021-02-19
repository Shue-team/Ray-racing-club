//
// Created by arseny on 06.02.2021.
//

// You may need to build the project (run Qt uic code generator) to get "ui_MainWindow.h" resolved

#include <QAction>

#include "MainWindow.h"
#include "ui_MainWindow.h"

#include "Renderer.h"

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
    //TODO (3): Написать вызов метода-обёртки класса Renderer, преобразовать его результат в QImage
    // и добавить в ui->imageDisplay.
}
