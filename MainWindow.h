//
// Created by arseny on 06.02.2021.
//

#ifndef RAY_RACING_CLUB_MAINWINDOW_H
#define RAY_RACING_CLUB_MAINWINDOW_H

#include <QWidget>
#include "Renderer.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QWidget {
Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);

    void paintEvent(QPaintEvent *event) override;

    ~MainWindow() override;

private:
    Ui::MainWindow* mUi;
    Renderer* mRenderer;
    Camera*  mCamera;
private slots:
    //void onRenderAction();
    void moveLeft();
    void moveRight();
    void moveDown();
    void moveUp();
    void zoomIn();
    void zoomOut();
    void rotateUp();
    void rotateDown();
    void rotateLeft();
    void rotateRight();
};

#endif //RAY_RACING_CLUB_MAINWINDOW_H
