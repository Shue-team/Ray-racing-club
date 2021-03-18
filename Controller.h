#include <QObject>
#include "Camera.h"


/*
 * wasd - moving in Oxy plane
 * arrows - rotating camera
 * e - zoom in
 * q - zoom out
*/
class Controller:public QObject {
public:
    Controller(Camera* cam, QObject* parent = nullptr);
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
    float mouseAcc = 0.001f;
    Camera* mCam;
    QPoint mLastMousePos;
};