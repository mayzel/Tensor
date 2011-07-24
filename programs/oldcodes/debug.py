# coding: utf-8

#デバッグ用関数群


import datetime


#ログを取る
class Logger:

    def __init__(self,name):
        self.name = name
        d = datetime.datetime.today()

        self.filename = "log/" + name + '%s.%s%s.%s%s.log' % (d.year, d.month, d.day, d.hour, d.minute)
        self.filestream = open(self.filename, 'w')

    def __del__(self):
        self.filestream.close()

    def WriteLine(self,message):
        self.Write(str(message))
        self.Write("\n")


    def Write(self,message):
        self.filestream.write(str(message))


    def Release(self):
        self.filestream.close()


def testlogger():
    logger = Logger("testlog")
    for i in xrange(1000):
        s = str(pow(i,4))
        logger.WriteLine(s)

    logger.Release()

