#pragma once
#include <QMainWindow>
#include <QPushButton>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/parameter.hpp>
#include <rcl_interfaces/msg/parameter.hpp>
#include <rcl_interfaces/srv/set_parameters.hpp>
#include <thread>

#include "FaceView.h"
#include "UserSelectorDialog.h"


class QComboBox;
class QPushButton;

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

private slots:
    void onUserButtonClicked();

private:
    QPushButton* m_userButton;
    FaceView* m_faceView;
    UserSelectorDialog* m_userSelectorDialog;


    rclcpp::Node::SharedPtr node_;


    std::thread ros_spin_thread_;
};
