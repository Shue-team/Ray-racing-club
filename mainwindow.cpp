//
// Created by arseny on 06.02.2021.
//

// You may need to build the project (run Qt uic code generator) to get "ui_MainWindow.h" resolved

#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget* parent) :
        QWidget(parent), mUi(new Ui::MainWindow) {
    mUi->setupUi(this);

}

MainWindow::~MainWindow() {
    delete mUi;
}
