import datetime
import os
import shutil
def removeFolderBeforeAHour():
    now = datetime.datetime.now()
    hour = str(now.hour - 1)
    folder_path = "temp/" + hour
    # folder_path = "C:\\Users\\Administrator\\Documents" + "\\" + hour
    if os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
        print("folder removed")
    else:
        print("folder not found")
removeFolderBeforeAHour()