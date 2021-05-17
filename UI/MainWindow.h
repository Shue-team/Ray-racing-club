//
// Created by arseny on 06.02.2021.
//

#ifndef RAY_RACING_CLUB_MAINWINDOW_H
#define RAY_RACING_CLUB_MAINWINDOW_H

#include <QWidget>
#include <QGraphicsScene>

#include "../Render/Renderer.h"
#include "WorldEdit.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QWidget {
Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);

    ~MainWindow() override;

private:
    void initWorldEdit();

    Ui::MainWindow* mUi;
    Renderer* mRenderer;
    WorldEdit* mWorldEdit;

private slots:
    void onWorldEditButton();
    void onRenderAction() const;
};

#endif //RAY_RACING_CLUB_MAINWINDOW_H
