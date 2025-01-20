from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton
import subprocess


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("分离逻辑示例")
        layout = QVBoxLayout()
        self.label = QLabel("点击按钮计数: 0")
        layout.addWidget(self.label)
        self.button = QPushButton("点击我")
        self.button.clicked.connect(self.button_click)
        layout.addWidget(self.button)

        # 添加一个新的按钮用于调用命令行
        self.cmd_button = QPushButton("调用命令行")
        self.cmd_button.clicked.connect(self.run_command)
        layout.addWidget(self.cmd_button)

        self.setLayout(layout)

    def button_click(self):
        current_count = int(self.label.text().split(': ')[1])
        new_count = current_count + 1
        self.label.setText(f"点击按钮计数: {new_count}")

    def run_command(self):
        try:
            # 这里可以修改为你想要执行的命令行命令，例如 'ls' （在 Linux 或 macOS 上）或 'dir' （在 Windows 上）
            result = subprocess.run(['ls'], capture_output=True, text=True)
            print("命令输出：")
            print(result.stdout)
            if result.stderr:
                print("错误信息：")
                print(result.stderr)
        except Exception as e:
            print(f"执行命令时出错：{e}")


def main():
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()