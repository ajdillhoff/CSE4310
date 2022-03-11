import sys
import random
from PySide2 import QtCore, QtWidgets, QtGui
from skvideo.io import vread

class QtDemo(QtWidgets.QWidget):
    def __init__(self, frames):
        super().__init__()

        self.frames = frames

        self.current_frame = 0

        self.button = QtWidgets.QPushButton("Next Frame")

        # Configure image label
        self.img_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        h, w, _ = self.frames[0].shape
        img = QtGui.QImage(self.frames[0], w, h, QtGui.QImage.Format_RGB888)
        self.img_label.setPixmap(QtGui.QPixmap.fromImage(img))

        # Configure slider
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.frame_slider.setTickInterval(1)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(self.frames.shape[0]-1)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.img_label)
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.frame_slider)

        # Connect functions
        self.button.clicked.connect(self.on_click)
        self.frame_slider.sliderMoved.connect(self.on_move)

    @QtCore.Slot()
    def on_click(self):
        if self.current_frame == self.frames.shape[0]-1:
            return
        h, w, _ = self.frames[self.current_frame].shape
        img = QtGui.QImage(self.frames[self.current_frame], w, h, QtGui.QImage.Format_RGB888)
        self.img_label.setPixmap(QtGui.QPixmap.fromImage(img))
        self.current_frame += 1

    @QtCore.Slot()
    def on_move(self, pos):
        self.current_frame = pos
        h, w, _ = self.frames[self.current_frame].shape
        img = QtGui.QImage(self.frames[self.current_frame], w, h, QtGui.QImage.Format_RGB888)
        self.img_label.setPixmap(QtGui.QPixmap.fromImage(img))


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("USAGE: qtdemo.py PATH_TO_VIDEO")
        sys.exit(1)

    frames = vread(sys.argv[1])

    app = QtWidgets.QApplication([])

    widget = QtDemo(frames)
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec_())
