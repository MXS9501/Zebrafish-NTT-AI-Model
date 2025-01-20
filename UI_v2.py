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

        # 添加运行另一个.py文件的按钮
        self.run_file_button = QPushButton("运行另一个文件")
        self.run_file_button.clicked.connect(self.run_another_file)
        layout.addWidget(self.run_file_button)

        self.setLayout(layout)

    def button_click(self):
        current_count = int(self.label.text().split(': ')[1])
        new_count = current_count + 1
        self.label.setText(f"点击按钮计数: {new_count}")

    def run_another_file(self):
        try:
            # 将 'another_file.py' 替换为你实际要运行的.py文件名
            subprocess.Popen(['python', 'another_file.py'])
        except Exception as e:
            print(f"运行文件时出错: {e}")


def main():
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()