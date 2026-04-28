#include <QApplication>
#include "widgets/MainWindow.h"

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    QApplication app(argc, argv);
    MainWindow w;
    w.show();
    int result = app.exec();
    rclcpp::shutdown();
    return result;
}
