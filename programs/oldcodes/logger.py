# coding: utf-8

#デバッグ用関数群

import datetime


#ログを取る
class Logger:
    def printline(self,message):
        print "#", message

    def __init__(self,name):
        self.flushInterval = datetime.timedelta(0,30,0)
        self.disp = True
        self.name = name
        d = datetime.datetime.today()

        self.filename = "log/" + name + '%04d.%02d%02d.%02d%02d.%02d.log' % (d.year, d.month, d.day, d.hour, d.minute, d.second)
        self.filestream = open(self.filename, 'w')

        self.lastUpdate = d

    def __del__(self):
        self.filestream.close()

    def WriteLine(self,message="",disp=True):
        self.Write(str(message) + "\n",disp)


    def Write(self,message="",disp=True):
        self.filestream.write(str(message))
        if disp:
            self.printline(str(message).rstrip())
        now = datetime.datetime.today()
        self.filestream.flush()
        if now > self.lastUpdate + self.flushInterval:
            self.filestream.flush()
            self.lastUpdate = now

    def Release(self):
        self.filestream.close()

def testlogger():
    logger = Logger("testlog")
    for i in xrange(1000):
        s = str(pow(i,4))
        logger.WriteLine(s)

    logger.Release()

