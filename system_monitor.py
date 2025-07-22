import sys
import psutil
import subprocess
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QTimer


class SystemMonitor(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Mac システムモニター")
        self.setFixedSize(300, 150)

        # ラベルの作成
        self.cpu_label = QLabel("CPU 使用率: ", self)
        self.mem_label = QLabel("メモリ使用率: ", self)
        self.temp_label = QLabel("CPU 温度: ", self)

        # レイアウト
        layout = QVBoxLayout()
        layout.addWidget(self.cpu_label)
        layout.addWidget(self.mem_label)
        layout.addWidget(self.temp_label)
        self.setLayout(layout)

        # タイマーで更新
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_stats)
        self.timer.start(1000)  # 1秒ごとに更新

    def update_stats(self):
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        temp = self.get_cpu_temp()

        self.cpu_label.setText(f"CPU 使用率: {cpu:.1f}%")
        self.mem_label.setText(f"メモリ使用率: {mem:.1f}%")
        self.temp_label.setText(f"CPU 温度: {temp}")

    def get_cpu_temp(self):
        try:
            result = subprocess.check_output(["osx-cpu-temp"])
            return result.decode("utf-8").strip()
        except Exception as e:
            return f"エラー: {e}"


if __name__ == "__main__":
    app = QApplication(sys.argv)
    monitor = SystemMonitor()
    monitor.show()
    sys.exit(app.exec_())
