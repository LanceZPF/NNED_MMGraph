import os
import sys
import PyQt5 as Qt
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtCore import QMimeData, QThread, pyqtSignal, QSize
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QAction, QFileDialog, QMessageBox,\
                            QFontDialog, QColorDialog
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QDialog, QLabel, QLineEdit, QPushButton, \
    QGridLayout, QVBoxLayout, QHBoxLayout, QMessageBox
from PyQt5.QtWidgets import QApplication, QWidget, QProgressBar, QPushButton, QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QTextEdit, QTextBrowser, QHBoxLayout, QVBoxLayout

from UI_used import *

class Demo(QMainWindow):
    def __init__(self):
        super(Demo, self).__init__()
        self.resize(500, 360)  # 设置窗体大小
        self.mode_menu = self.menuBar().addMenu('功能选择')
        self.about_menu = self.menuBar().addMenu('关于')

        self.single_action = QAction('单文本处理', self)
        self.group_action = QAction('批次处理', self)
        self.about_action = QAction('关于我们', self)



        self.single_page = single_process() # 单文本处理
        self.group_page = group_process()  #批量处理
        self.about_page = about_page()


        self.menu_init()
        self.action_init()

    def layout_init(self):
        self.grid.addWidget(self.image,2,2)
        self.setLayout(self.grid)

    def menu_init(self):
        self.mode_menu.addAction(self.single_action)
        self.mode_menu.addAction(self.group_action)
        self.about_menu.addAction(self.about_action)

    def action_init(self):
        self.single_action.triggered.connect(self.change_to_single)
        self.group_action.triggered.connect(self.change_to_group)
        self.about_action.triggered.connect(self.change_to_about)

    def change_to_single(self):
        self.single_page.exec_()

    def change_to_group(self):
        self.group_page.exec_()

    def change_to_about(self):
        self.about_page.exec_()


text = ''


class single_process(QDialog):
    def __init__(self):
        super(single_process, self).__init__()
        self.resize(500, 360)
        self.edit_label = QLabel('输入文本', self)
        self.result_label = QLabel('消歧结果（实体名称、偏移量、链接）', self)
        self.text_edit = QTextEdit(self)
        self.text_browser = QTextBrowser(self)
        self.pushbotton = QPushButton('开始消歧',self)



        self.grid_layout1 = QGridLayout()
        self.v1 = QVBoxLayout()
        self.v2 = QVBoxLayout()
        self.h = QHBoxLayout()

        self.layout_init()
        self.connect_init()


    def connect_init(self):
        self.text_edit.setPlaceholderText('请在此输入消歧的话')
        self.pushbotton.setEnabled(False)
        self.text_edit.textChanged.connect(self.check_input_func)
        self.pushbotton.clicked.connect(self.single_model_process)


    def layout_init(self):
        self.v1.addWidget(self.edit_label)
        self.v1.addWidget(self.text_edit)
        self.v2.addWidget(self.result_label)
        self.v2.addWidget(self.text_browser)
        self.h.addLayout(self.v1)
        self.h.addWidget(self.pushbotton)
        self.h.addLayout(self.v2)
        self.setLayout(self.h)

    def check_input_func(self):
        if self.text_edit.toPlainText():
            self.pushbotton.setEnabled(True)
        else:
            self.pushbotton.setEnabled(False)


    def single_model_process(self):
        # 用于将单文本送入模型进行训练
        self.Thread = MyThread(self.text_edit.toPlainText())
        self.Thread.my_signal.connect(self.print_result)
        self.Thread.start()


    def print_result(self, adict):
        if len(adict['entity_name'])==0:
            self.text_browser.setText('不存在待消歧的实体')
        else:
            self.text_browser.setText('{}     {}     {}'.format(adict['entity_name'][0], adict['offset'][0], adict['entity_id'][0]))
        # for i in range(len(adict['offset'])):
        #     print('{} {} {}'.format(adict['entity_name'][i], adict['offset'][i], adict['entity_id'][i]))





class group_process(QDialog):
    def __init__(self):
        super(group_process, self).__init__()
        self.resize(500, 360)

        # 按钮设置
        self.file_botton = QPushButton('选择文件',self)
        self.process_botton = QPushButton('开始消歧',self)


        #进度条设置
        self.progressbar = QProgressBar(self)
        self.progressbar.setMinimum(0)
        self.progressbar.setMaximum(100)

        # 消息框设置
        self.messageBox1 = QMessageBox(QMessageBox.Question,'温馨提示：','请注意你的文件输入格式，如果不正确，将会产生错误结果！想清楚了吗？')
        # self.messageBox1.addButton(QPushButton('想清楚了'), QPushButton('查看文件格式'), QPushButton('放弃'))
        self.yes = self.messageBox1.addButton('继续选择文件', QMessageBox.YesRole)
        self.no = self.messageBox1.addButton('取消', QMessageBox.NoRole)
        self.thinking = self.messageBox1.addButton('查看文件格式',QMessageBox.NoRole)
        self.yes.clicked.connect(self.chooseFile)
        self.thinking.clicked.connect(self.showMessageBox)


        self.cwd = os.getcwd()
        self.filename = 1


        self.v = QVBoxLayout()
        self.h = QHBoxLayout()

        self.layout_init()
        self.connect_init()
    def layout_init(self):
        self.h.addWidget(self.file_botton)
        self.h.addWidget(self.process_botton)
        self.v.addLayout(self.h)
        self.v.addWidget(self.progressbar)
        self.setLayout(self.v)

    def connect_init(self):
        self.file_botton.clicked.connect(self.file_botton_messagebox)
        self.process_botton.clicked.connect(self.startED)
    #
    # def show_messagebox1(self):
    #     QMessageBox.information(self,'消歧成功','恭喜你，消歧结果已保存到当前目录的下！')

    def file_botton_messagebox(self):
        self.messageBox1.show()
        if self.messageBox1.clickedButton() == self.yes:
            #继续选择
            pass
        elif self.messageBox1.clickedButton() == self.no:
            pass
        elif self.messageBox1.clickedButton() == self.thinking:
            pass
    # def show_messagebox(self):
    #     choice = QMessageBox.information(self, '温馨提示', '请注意你的文件输入格式，如果不正确，将会产生错误结果！想清楚了吗？',QMessageBox.Yes | QMessageBox.No)
    #     if choice == QMessageBox.Yes:
    #         # 继续选择文件
    #         pass
    #     elif choice == QMessageBox.No:
    #         # 退出
    #         pass

    def chooseFile(self):
        file_choose = QFileDialog.getOpenFileName(self,"选取文件", self.cwd)
        self.filename = file_choose[0]
    def showMessageBox(self):
        QMessageBox.information(self, '文本格式', '0\t我想吃爆米花\n1\t中国银行的股票跌了\n...')

    def startED(self):
        self.Thread = MyThread_Group(self.filename)
        self.Thread.my_signal.connect(self.setbar)
        self.Thread.start()

    def setbar(self,value):
        self.progressbar.setValue(value)

class about_page(QDialog):
    def __init__(self):
        super(about_page, self).__init__()
        scale = 0.8
        # self.resize(500, 360)
        self.img = QImage('./img/logo.png')
        self.size = QSize(200,200)
        self.jpg = QPixmap.fromImage(self.img.scaled(self.size))
        self.image = QLabel(self)
        # self.jpg = QPixmap('./img/杜兰特.jpg')
        self.image.setPixmap(self.jpg)
        self.image.resize(200,200)
        self.teamname = QLabel(self)
        self.teamname.setText('队名：想吃海底捞')

        self.v = QVBoxLayout()
        self.grid = QGridLayout()

        self.layout_init()
    def layout_init(self):
        # self.v.addWidget(self.image)
        # self.v.addWidget(self.teamname)
        self.grid.addWidget(self.image, 2,2)
        self.grid.addWidget(self.teamname,3,2)
        self.setLayout(self.grid)



class MyThread(QThread):
    my_signal = pyqtSignal(dict)
    def __init__(self,text):
        super(MyThread, self).__init__()
        self.text = text
    def run(self):
        adict = UI_Single_Process(self.text)
        self.my_signal.emit(adict)
class MyThread_Group(QThread):
    my_signal = pyqtSignal(int)
    def __init__(self, filename):
        super(MyThread_Group, self).__init__()
        self.filename = filename
    def run(self):
        filename  = self.filename
        # 直接集成，生存所有结果。
        # 只需修改注释处的代码即可
        UI_get_Result_input(filename)
        train_loader = DataLoader(UI_EL_Result_datasets(), batch_size=16, shuffle=True)  # 改数据集类
        model = gcn_bert0(t=0.4, adj_file='data/god_adj.pkl')  # 改模型名称
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load('./ELmodel/GCN_Supervised.ckpt'))  # 改模型地址

        model.eval().cuda()
        predict_all = np.array([], dtype=int)
        confidence_all = np.array([], dtype=float)
        index = {'text_id': [], 'offset': [], 'entity_id': []}
        with torch.no_grad():
            for i, sample in enumerate(tqdm(train_loader)):
                self.my_signal.emit(i/len(train_loader)*100)
                # if i==1:
                #         break
                index['text_id'] += sample[19].numpy().tolist()  # 句子位置
                index['offset'] += sample[20].numpy().tolist()  # 偏移量
                index['entity_id'] += list(sample[21])  # id
                outputs = model(sample).float()
                confidence, predic = torch.max(outputs, dim=1)  # 增加了一个confidence
                predic = predic.cpu().numpy()
                confidence = confidence.cpu().numpy()
                predict_all = np.append(predict_all, predic)
                confidence_all = np.append(confidence_all, confidence)
        np.save('intermediate/predict.npy', predict_all)  # 两者通过索引进行对应
        np.save('intermediate/confidence.npy', confidence_all)
        with open('intermediate/index.json', 'w') as f:
            json.dump(index, f)
        process()
        sort()
        QMessageBox.information(self, '消歧成功',
                                '恭喜你，消歧结果的路径是{}'.format('/home/lance/Desktop/ED/result' + 'UI_result.json'))



if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = Demo()
    demo.show()
    sys.exit(app.exec_())