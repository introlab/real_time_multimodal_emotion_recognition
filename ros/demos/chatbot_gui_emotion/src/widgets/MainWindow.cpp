#include "MainWindow.h"
#include <QComboBox>
#include <QPushButton>
#include <QVBoxLayout>
#include <QWidget>
#include <QTimer>

template<typename T>
static rcl_interfaces::msg::Parameter create_param(const std::string& name, const T& value)
{
    rclcpp::Parameter p(name, value);
    return p.to_parameter_msg();
}

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent)
{
    node_ = rclcpp::Node::make_shared("chat_param_setter");

    ros_spin_thread_ = std::thread([this]() { rclcpp::spin(node_); });

    QWidget* centralWidget = new QWidget(this);
    QVBoxLayout* layout = new QVBoxLayout(centralWidget);

    m_userButton = new QPushButton("", centralWidget);
    m_userButton->setText("Changer d'utilisateur");
    m_userButton->setMinimumHeight(50);
    const int font_size = 18;
    QFont font = m_userButton->font();
    font.setPointSize(font_size);
    m_userButton->setFont(font);
    layout->addWidget(m_userButton);
    m_userButton->setEnabled(false);
    // Set Background translucent
    m_userButton->setStyleSheet("background-color: rgba(255, 255, 255, 255);");
    connect(m_userButton, &QPushButton::clicked, this, &MainWindow::onUserButtonClicked);


    m_faceView = new FaceView(*node_, centralWidget);
    layout->addWidget(m_faceView);
    centralWidget->setLayout(layout);
    setCentralWidget(centralWidget);
    setWindowTitle("Chatbot GUI Move");
    // resize(400, 200);
    showFullScreen();

    m_userSelectorDialog = new UserSelectorDialog(node_, this);
    m_userSelectorDialog->setModal(true);
    m_userSelectorDialog->exec();
    m_userButton->setEnabled(true);
    m_userButton->setText(m_userSelectorDialog->getSelectedParticipantName());
}

MainWindow::~MainWindow()
{
    rclcpp::shutdown();
    if (ros_spin_thread_.joinable())
        ros_spin_thread_.join();
}

void MainWindow::onUserButtonClicked()
{
    m_userSelectorDialog->setModal(true);
    m_userSelectorDialog->exec();
    m_userButton->setText(m_userSelectorDialog->getSelectedParticipantName());
}
