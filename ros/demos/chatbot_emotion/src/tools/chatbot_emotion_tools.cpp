#include <chatbot_emotion/tools/chatbot_emotion_tools.hpp>

#include <string>
#include <memory>
#include <nlohmann/json.hpp>


using json = nlohmann::json;
using namespace std;


ChatbotMoveTools::ChatbotMoveTools(shared_ptr<rclcpp::Node> node, rclcpp::CallbackGroup::SharedPtr callbackGroup)
    : node_(std::move(node)),
      callbackGroup_(callbackGroup)
{
    service_get_activity_ = node_->create_service<behavior_srvs::srv::ChatToolsFunctionCall>(
        "/chat/tools/functions/get_activity_data",
        std::bind(
            &ChatbotMoveTools::handle_get_activity_data_request,
            this,
            std::placeholders::_1,
            std::placeholders::_2),
        rmw_qos_profile_services_default,
        callbackGroup_);

    activity_client_ =
        node_->create_client<opentera_link_srvs::srv::ActivityMonitoring>("opentera_link/activity_monitoring");

    service_get_activity_schedule_ = node_->create_service<behavior_srvs::srv::ChatToolsFunctionCall>(
        "/chat/tools/functions/get_activity_schedule",
        std::bind(
            &ChatbotMoveTools::handle_get_activity_schedule_request,
            this,
            std::placeholders::_1,
            std::placeholders::_2),
        rmw_qos_profile_services_default,
        callbackGroup_);

    activity_schedule_client_ =
        node_->create_client<opentera_link_srvs::srv::ActivityScheduling>("opentera_link/activity_scheduling");
}

void ChatbotMoveTools::handle_get_activity_data_request(
    const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Request> request,
    const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Response> response)
{
    RCLCPP_INFO(node_->get_logger(), "Received get_activity request");

    uint64_t relative_day = 0;
    std::string participant_uuid;
    try
    {
        if (request->function_arguments.length() > 0)
        {
            nlohmann::json args = nlohmann::json::parse(request->function_arguments);
            if (args.contains("relative_day"))
            {
                relative_day = args["relative_day"].get<uint64_t>();
            }

            if (args.contains("participant_uuid"))
            {
                participant_uuid = args["participant_uuid"].get<std::string>();
            }
        }
    }
    catch (const std::exception& e)
    {
        RCLCPP_WARN(node_->get_logger(), "Failed to parse arguments: %s", e.what());
    }

    RCLCPP_INFO(node_->get_logger(), "Processing activity data for relative day: %ld", relative_day);

    if (!activity_client_->wait_for_service(std::chrono::seconds(10)))
    {
        RCLCPP_ERROR(node_->get_logger(), "Activity data service unavailable");
        response->ok = false;
        response->result = "{\"error\": \"Activity service unavailable\"}";
        return;
    }

    try
    {
        auto req = std::make_shared<opentera_link_srvs::srv::ActivityMonitoring::Request>();
        req->relative_day = relative_day;
        req->participant_uuid = participant_uuid;

        std::promise<std::shared_ptr<opentera_link_srvs::srv::ActivityMonitoring::Response>> promise;
        std::future<std::shared_ptr<opentera_link_srvs::srv::ActivityMonitoring::Response>> future =
            promise.get_future();

        auto callback =
            [&promise](rclcpp::Client<opentera_link_srvs::srv::ActivityMonitoring>::SharedFuture inner_future)
        { promise.set_value(inner_future.get()); };

        activity_client_->async_send_request(req, callback);

        auto status = future.wait_for(std::chrono::seconds(5));

        if (status != std::future_status::ready)
        {
            RCLCPP_ERROR(node_->get_logger(), "Timeout waiting for activity response");
            response->ok = false;
            response->result = "{\"error\": \"Timeout\"}";
            return;
        }

        auto res = future.get();
        if (res)
        {
            // Directly return the JSON string from the response
            response->ok = true;
            response->result = res->actimetry_data_json;
        }
    }
    catch (const std::exception& e)
    {
        RCLCPP_ERROR(node_->get_logger(), "Exception in Activity service callback: %s", e.what());
        response->ok = false;
        response->result = "{\"error\": \"Internal error: " + std::string(e.what()) + "\"}";
    }
}

void ChatbotMoveTools::handle_get_activity_schedule_request(
    const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Request> request,
    const std::shared_ptr<behavior_srvs::srv::ChatToolsFunctionCall::Response> response)
{
    RCLCPP_INFO(node_->get_logger(), "Received get_activity_schedule request");

    string relative_day;
    try
    {
        if (request->function_arguments.length() > 0)
        {
            nlohmann::json args = nlohmann::json::parse(request->function_arguments);
            if (args.contains("relative_day"))
            {
                relative_day = args["relative_day"];
            }
        }
    }
    catch (const std::exception& e)
    {
        RCLCPP_WARN(node_->get_logger(), "Failed to parse arguments: %s", e.what());
    }

    RCLCPP_INFO(node_->get_logger(), "Processing activity data for relative day: %s", relative_day.c_str());

    if (!activity_schedule_client_->wait_for_service(std::chrono::seconds(10)))
    {
        RCLCPP_ERROR(node_->get_logger(), "Activity schedule service unavailable");
        response->ok = false;
        response->result = "{\"error\": \"Activity service unavailable\"}";
        return;
    }

    try
    {
        auto req = std::make_shared<opentera_link_srvs::srv::ActivityScheduling::Request>();
        req->relative_day = relative_day;

        std::promise<std::shared_ptr<opentera_link_srvs::srv::ActivityScheduling::Response>> promise;
        std::future<std::shared_ptr<opentera_link_srvs::srv::ActivityScheduling::Response>> future =
            promise.get_future();

        auto callback =
            [&promise](rclcpp::Client<opentera_link_srvs::srv::ActivityScheduling>::SharedFuture inner_future)
        { promise.set_value(inner_future.get()); };

        activity_schedule_client_->async_send_request(req, callback);

        auto status = future.wait_for(std::chrono::seconds(5));

        if (status != std::future_status::ready)
        {
            RCLCPP_ERROR(node_->get_logger(), "Timeout waiting for activity schedule response");
            response->ok = false;
            response->result = "{\"error\": \"Timeout\"}";
            return;
        }

        auto res = future.get();

        if (res)
        {
            response->ok = true;
            response->result = res->scheduled_activities;
        }
    }
    catch (const std::exception& e)
    {
        RCLCPP_ERROR(node_->get_logger(), "Exception in Activity service callback: %s", e.what());
        response->ok = false;
        response->result = "{\"error\": \"Internal error: " + std::string(e.what()) + "\"}";
    }
}
