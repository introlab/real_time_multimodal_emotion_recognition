#ifndef CHATBOT_MOVE__TOOLS__CHATBOT_MOVE_TOOLS_HPP
#define CHATBOT_MOVE__TOOLS__CHATBOT_MOVE_TOOLS_HPP

#include <rclcpp/rclcpp.hpp>
#include <behavior_srvs/srv/chat_tools_function_call.hpp>
#include <std_msgs/msg/u_int8.hpp>
#include <opentera_link_srvs/srv/activity_monitoring.hpp>
#include <opentera_link_srvs/srv/activity_scheduling.hpp>

class ChatbotMoveTools
{
public:
    ChatbotMoveTools(std::shared_ptr<rclcpp::Node> node, rclcpp::CallbackGroup::SharedPtr callbackGroup);
    ~ChatbotMoveTools() = default;

private:
    rclcpp::Node::SharedPtr node_;
    rclcpp::CallbackGroup::SharedPtr callbackGroup_;

    rclcpp::Service<behavior_srvs::srv::ChatToolsFunctionCall>::SharedPtr service_get_activity_;
    rclcpp::Service<behavior_srvs::srv::ChatToolsFunctionCall>::SharedPtr service_get_activity_schedule_;

    rclcpp::Client<opentera_link_srvs::srv::ActivityMonitoring>::SharedPtr activity_client_;
    rclcpp::Client<opentera_link_srvs::srv::ActivityScheduling>::SharedPtr activity_schedule_client_;

    void handle_get_activity_data_request(
        const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Request> request,
        const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Response> response);

    void handle_get_activity_schedule_request(
        const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Request> request,
        const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Response> response);
};

#endif
