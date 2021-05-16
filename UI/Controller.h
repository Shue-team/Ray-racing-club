#pragma once
#include <QObject>
#include <QEvent>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QEnterEvent>
#include "../Render/Camera.h"


/*
 * wasd - moving in Oxy plane
 * arrows - rotating camera
 * e - zoom in
 * q - zoom out
*/
class Controller:public QObject {
public:
    struct Steps {
        float moveStep;
        float rotStep;
        float zoomStep;
        float mouseAcc;
    };
    Controller(Camera* cam, Steps& steps, QObject* parent = nullptr);

    bool eventFilter(QObject* object, QEvent* event) override;
private:
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

    float mMoveStep = 0.1f;
    float mRotStep = 0.05f;
    float mZoomStep = 0.05f;
    float mMouseAcc = 0.001f;

    bool proceedKeyPress(QKeyEvent* keyEvent);
    bool proceedMouseMove(QMouseEvent* mouseEvent);
    bool proceedMouseButtons(QMouseEvent* mouseEvent);
    bool proceedWidgetEnter(QEnterEvent* enterEvent);

    Camera* mCam;
    QPoint mLastMousePos;
};