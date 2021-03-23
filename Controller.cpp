#include "Controller.h"

Controller::Controller(Camera* cam, Controller::Steps&& steps, QObject* parent):QObject(parent) {
    mCam = cam;
    mMoveStep = steps.moveStep;
    mRotStep = steps.rotStep;
    mZoomStep = steps.zoomStep;
    mMouseAcc = steps.mouseAcc;
}

bool Controller::eventFilter(QObject* object, QEvent* event) {
    switch (event->type()) {
    case QEvent::KeyPress:
        return proceedKeyPress((QKeyEvent*)event);

    case QEvent::MouseMove:
        return proceedMouseMove((QMouseEvent*)event);

    case QEvent::MouseButtonPress:
        return proceedMouseButtons((QMouseEvent*)event);

    case QEvent::Enter:
        return proceedWidgetEnter((QEnterEvent*)event);

    default:
        return false;
    }
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

bool Controller::proceedKeyPress(QKeyEvent* keyEvent) {
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

bool Controller::proceedMouseMove(QMouseEvent* mouseEvent) {
    bool fKeyPressed = mouseEvent->buttons() == Qt::MouseButton::LeftButton;
    float xRot = mMouseAcc * (mouseEvent->y() - mLastMousePos.y());
    float yRot = mMouseAcc * (mouseEvent->x() - mLastMousePos.x());
    
    if (!fKeyPressed) {
        xRot *= -1;
        yRot *= -1;
    }
    
    mCam->rotateOy(yRot);
    mCam->rotateOx(xRot);
    
    mLastMousePos = mouseEvent->pos();
    
    return true;
}

bool Controller::proceedMouseButtons(QMouseEvent* mouseEvent) {
    if (mouseEvent->button() == Qt::MouseButton::LeftButton) {
        mLastMousePos = mouseEvent->pos();
        return true;
    }
    return false;
}

bool Controller::proceedWidgetEnter(QEnterEvent* enterEvent) {
    mLastMousePos = enterEvent->pos();
    return true;
}

void Controller::zoomIn() {
    mCam->moveDepth(-mZoomStep);
}

void Controller::zoomOut() {
    mCam->moveDepth(mZoomStep);
}