//
// Created by awesyr on 06.03.2021.
//

#include <QImage>

#include "Renderer.h"

QImage Renderer::render(const Camera* camera, float& time) {
    uchar8* data = renderRaw(camera, time);
    return QImage(data, mImgWidth, mImgHeight, 3 * mImgWidth, QImage::Format_RGB888);
}