#pragma once


#include <QWidget>
#include <QLabel>
#include <QVBoxLayout>
#include <QTimer>
#include <QWebView>
#include <QPushButton>
#include <QComboBox>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/parameter.hpp>
#include <memory>
#include <utility>
#include <functional>
// #include <std_msgs/msg/int8.hpp>

class FaceView : public QWidget
{
    Q_OBJECT
    rclcpp::Node& m_node;
    // rclcpp::Publisher<std_msgs::msg::Int8>::SharedPtr chatServiceSWPub;


public:
    FaceView(rclcpp::Node& nodeHandle, QWidget* parent = nullptr);
    void setReadyCallback(std::function<void(void)> callback);

    // signals:
private slots:
    void avatarLoaded(bool ok);

private:
    void createUi();


    // UI members
    QWebView* m_avatarView;
    std::function<void(void)> m_callback;
};
