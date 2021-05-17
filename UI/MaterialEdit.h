#ifndef MATERIALEDIT_H
#define MATERIALEDIT_H

#include <QWidget>
#include "../Material/MaterialDef.h"

class MaterialEdit : public QWidget {
    Q_OBJECT
public:
    MaterialEdit(QWidget* parent = nullptr)
        : QWidget(parent) {}

    virtual MaterialDef* createDefinition()  const = 0;

signals:
    void dataFilled(bool flag);
};


#endif // MATERIALEDIT_H
