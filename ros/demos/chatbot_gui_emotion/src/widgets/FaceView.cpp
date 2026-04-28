#include "FaceView.h"


using namespace std;

constexpr const char* URL = "http://localhost:8080/face";
constexpr int RELOAD_INTERVAL_MS = 10000;

FaceView::FaceView(rclcpp::Node& node, QWidget* parent) : QWidget(parent), m_node(node)
{
    createUi();
}


void FaceView::avatarLoaded(bool ok)
{
    if (!ok)
    {
        QTimer::singleShot(RELOAD_INTERVAL_MS, m_avatarView, &QWebView::reload);
        return;
    }

    if (m_callback)
    {
        m_callback();  // This triggers the callback when the FaceView is ready
    }
}

void FaceView::setReadyCallback(std::function<void(void)> callback)
{
    m_callback = std::move(callback);  // Invoking the setReadyCallback method on the m_FaceView object
}


void FaceView::createUi()
{
    m_avatarView = new QWebView(this);
    m_avatarView->settings()->setAttribute(QWebSettings::JavascriptEnabled, true);
    m_avatarView->setUrl(QUrl(URL));
    m_avatarView->setZoomFactor(1.1);

    connect(m_avatarView, &QWebView::loadFinished, this, &FaceView::avatarLoaded);

    auto globalLayout = new QVBoxLayout;
    globalLayout->addWidget(m_avatarView);

    setLayout(globalLayout);
}
