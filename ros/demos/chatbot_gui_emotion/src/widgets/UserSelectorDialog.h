#pragma once

#include <QDialog>
#include <QComboBox>
#include <QPushButton>
#include <QVBoxLayout>
#include <QLabel>
#include <QTimer>
#include <rclcpp/rclcpp.hpp>
#include <opentera_link_srvs/srv/participant_context.hpp>
#include <opentera_link_srvs/srv/participant_names.hpp>
#include <opentera_link_msgs/msg/participant_name.hpp>

class UserSelectorDialog : public QDialog
{
    Q_OBJECT
public:
    explicit UserSelectorDialog(rclcpp::Node::SharedPtr node, QWidget* parent = nullptr);
    ~UserSelectorDialog() override = default;

public slots:

    QString getSelectedParticipantName() const { return participant_name; }
    QString getSelectedParticipantUUID() const { return participant_uuid; }

private slots:
    void onStartButtonToggled();
    void onLoadParticipantButtonToggled();
    void onParticipantChoice(const QString& text);

private:
    void add_participant_name();
    rclcpp::Node::SharedPtr m_node;
    rclcpp::Client<rcl_interfaces::srv::SetParameters>::SharedPtr param_client_chat_node_;
    rclcpp::Client<opentera_link_srvs::srv::ParticipantContext>::SharedPtr participant_context_client_;
    rclcpp::Client<opentera_link_srvs::srv::ParticipantNames>::SharedPtr participant_names_client_;
    rclcpp::Publisher<opentera_link_msgs::msg::ParticipantName>::SharedPtr participant_name_publisher_;

    QComboBox* participantComboBox;
    QPushButton* LoadParticipantButton;
    QPushButton* startButton;

    QString participant_name;
    QString participant_uuid;
};
