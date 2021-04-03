//
// Created by awesyr on 06.03.2021.
//

#include <QImage>

#include "Renderer.h"

QImage Renderer::render(const Camera* camera) {
    uchar8* data = renderRaw(camera);
    int bytesPerLine = 3 * mRi.imgWidth;
    return QImage(data, mRi.imgWidth, mRi.imgHeight, bytesPerLine, QImage::Format_RGB888);
}