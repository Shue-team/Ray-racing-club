#include <QEvent>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QEnterEvent>
#include "Controller.h"

Controller::Controller(Camera* cam, QObject* parent) :QObject(parent) {
    mCam = cam;
}

bool Controller::eventFilter(QObject* object, QEvent* event) {
    if (event->type() == QEvent::KeyPress) {
        QKeyEvent* keyEvent = (QKeyEvent*)event;
        if (keyEvent->modifiers() != Qt::NoModifier) {
            return false;
        }
        switch (keyEvent->key()) {
        case Qt::Key_W:
            moveUp();
            return true;
        case Qt::Key_S:
            moveDown();
            return true;
        case Qt::Key_A:
            moveLeft();
            return true;
        case Qt::Key_D:
            moveRight();
            return true;
        case Qt::Key_Up:
            rotateUp();
            return true;
        case Qt::Key_Down:
            rotateDown();
            return true;
        case Qt::Key_Left:
            rotateLeft();
            return true;
        case Qt::Key_Right:
            rotateRight();
            return true;
        case Qt::Key_E:
            zoomIn();
            return true;
        case Qt::Key_Q:
            zoomOut();
            return true;
        default:
            return false;
        }
    }
    else if (event->type() == QEvent::MouseMove) {
        QMouseEvent* mouseEvent = (QMouseEvent*)event;
        bool fKeyPressed = mouseEvent->buttons() == Qt::MouseButton::LeftButton;
        float xRot = mouseAcc * (mouseEvent->y() - mLastMousePos.y());
        float yRot = mouseAcc * (mouseEvent->x() - mLastMousePos.x());
        if (!fKeyPressed) {
            xRot *= -1;
            yRot *= -1;
        }
        mCam->rotateOy(yRot);
        mCam->rotateOx(xRot);
        mLastMousePos = mouseEvent->pos();
    }
    else if (event->type() == QEvent::MouseButtonPress) {
        QMouseEvent* mouseEvent = (QMouseEvent*)event;
        if (mouseEvent->button() == Qt::MouseButton::LeftButton) {
            mLastMousePos = mouseEvent->pos();
        }
    }
    else if (event->type() == QEvent::Enter) {
        QEnterEvent* enterEvent = (QEnterEvent*) event;
        mLastMousePos = enterEvent->pos();
    }
    return false;
}

void Controller::moveLeft() {
    mCam->moveHorz(-mMoveStep);
}

void Controller::moveRight() {
    mCam->moveHorz(mMoveStep);
}

void Controller::moveDown() {
    mCam->moveVert(-mMoveStep);
}

void Controller::moveUp() {
    mCam->moveVert(mMoveStep);
}

void Controller::rotateUp() {
    mCam->rotateOx(mRotStep);
}

void Controller::rotateDown() {
    mCam->rotateOx(-mRotStep);
}

void Controller::rotateLeft() {
    mCam->rotateOy(mRotStep);
}

void Controller::rotateRight() {
    mCam->rotateOy(-mRotStep);
}

void Controller::zoomIn() {
    mCam->moveDepth(-mZoomStep);
}

void Controller::zoomOut() {
    mCam->moveDepth(mZoomStep);
}