#include "UserSelectorDialog.h"

using namespace std;

UserSelectorDialog::UserSelectorDialog(rclcpp::Node::SharedPtr node, QWidget* parent) : QDialog(parent), m_node(node)
{
    param_client_chat_node_ = m_node->create_client<rcl_interfaces::srv::SetParameters>("/chat_node/set_parameters");

    participant_context_client_ =
        m_node->create_client<opentera_link_srvs::srv::ParticipantContext>("/opentera_link/participant_context");
    participant_names_client_ =
        m_node->create_client<opentera_link_srvs::srv::ParticipantNames>("/opentera_link/participant_names");
    participant_name_publisher_ = m_node->create_publisher<opentera_link_msgs::msg::ParticipantName>(
        "/opentera_link/participant_name",
        rclcpp::QoS(10));

    setWindowTitle("Sélectionner un utilisateur");


    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    setLayout(mainLayout);

    QWidget* centralWidget = new QWidget(this);
    QVBoxLayout* layout = new QVBoxLayout(centralWidget);

    participantComboBox = new QComboBox(centralWidget);
    participantComboBox->setEditable(false);

    // Set Font size to 18
    const int font_size = 18;
    QFont font = participantComboBox->font();
    font.setPointSize(font_size);
    participantComboBox->setFont(font);

    participantComboBox->setMinimumHeight(50);
    layout->addWidget(participantComboBox);


    LoadParticipantButton = new QPushButton("Charger les participants", centralWidget);
    LoadParticipantButton->setMinimumHeight(50);

    font = LoadParticipantButton->font();
    font.setPointSize(font_size);
    LoadParticipantButton->setFont(font);
    // Do not display the button
    LoadParticipantButton->hide();
    layout->addWidget(LoadParticipantButton);


    startButton = new QPushButton("Démarrer", centralWidget);
    startButton->setMinimumHeight(50);
    // Set background color to light green
    startButton->setStyleSheet("background-color: lightgreen");

    font = startButton->font();
    font.setPointSize(font_size);
    startButton->setFont(font);
    startButton->setEnabled(false);
    layout->addWidget(startButton);

    centralWidget->setLayout(layout);
    mainLayout->addWidget(centralWidget);

    setMinimumSize(500, 500);


    connect(startButton, &QPushButton::clicked, this, &UserSelectorDialog::onStartButtonToggled);
    connect(participantComboBox, &QComboBox::currentTextChanged, this, &UserSelectorDialog::onParticipantChoice);
    connect(LoadParticipantButton, &QPushButton::clicked, this, &UserSelectorDialog::onLoadParticipantButtonToggled);

    // Call the next function after 5 seconds to load participants
    QTimer::singleShot(5000, this, &UserSelectorDialog::onLoadParticipantButtonToggled);
}

void UserSelectorDialog::onStartButtonToggled()
{
    opentera_link_msgs::msg::ParticipantName participant_name_msg;
    participant_name_msg.participant_name = participant_name.toStdString();
    participant_name_msg.participant_uuid = participant_uuid.toStdString();
    participant_name_publisher_->publish(participant_name_msg);
    // Close the dialog
    accept();
}

void UserSelectorDialog::onLoadParticipantButtonToggled()
{
    participantComboBox->clear();
    participant_name = "";
    participant_uuid = "";
    startButton->setEnabled(false);
    add_participant_name();
}

void UserSelectorDialog::onParticipantChoice(const QString&)
{
    int idx = participantComboBox->currentIndex();

    QString display = participantComboBox->currentText();
    QString uuid = participantComboBox->itemData(idx).toString();

    // Extract name (everything before " - ")
    QString name = display.section(" - ", 0, 0);

    // Now you have both name and uuid
    participant_name = name;
    participant_uuid = uuid;
}

void UserSelectorDialog::add_participant_name()
{
    using Service = opentera_link_srvs::srv::ParticipantNames;

    auto req = std::make_shared<Service::Request>();
    auto future = participant_names_client_->async_send_request(req);

    auto ret = future.wait_for(std::chrono::seconds(5));
    if (ret != std::future_status::ready)
    {
        RCLCPP_ERROR(m_node->get_logger(), "Timeout waiting for ParticipantContext service response");
        QTimer::singleShot(2000, this, &UserSelectorDialog::onLoadParticipantButtonToggled);
        return;
    }

    auto res = future.get();

    if (res->participant_names.size() == 0)
    {
        RCLCPP_ERROR(m_node->get_logger(), "No participants found");
        QTimer::singleShot(2000, this, &UserSelectorDialog::onLoadParticipantButtonToggled);
        return;
    }

    startButton->setEnabled(true);

    for (size_t i = 0; i < res->participant_names.size(); ++i)
    {
        QString display = QString::fromStdString(
            res->participant_names[i]);  //+" - " + QString::fromStdString(res->participant_uuids[i]);
        participantComboBox->addItem(display, QString::fromStdString(res->participant_uuids[i]));
    }
}
