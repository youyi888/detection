from PyQt5.QtWidgets import QApplication, QFileDialog
import sys

app = QApplication(sys.argv)
file_dict = {}  # 存储文件夹路径的字典

# 打开文件夹选择对话框，获取选择的文件夹路径列表
dialog = QFileDialog()
dialog.setFileMode(QFileDialog.Directory)
dialog.setOption(QFileDialog.ShowDirsOnly)
dialog.setOption(QFileDialog.DontUseNativeDialog)
dialog.setOption(QFileDialog.ReadOnly)
dialog.setOption(QFileDialog.DontUseCustomDirectoryIcons)

if dialog.exec():
    folders = dialog.selectedFiles()

# 将文件夹路径存储到字典中
for folder in folders:
    folder_name = dialog.directory().relativeFilePath(folder)  # 获取文件夹名称
    file_dict[folder_name] = folder 
    print(folder )

