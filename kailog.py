import os


class KaiLogger:
    path = ""
    rotating = False
    datetime = False

    def __init__(self, path, rotating=False, max_bytes=0, backupCount=0):
        self.path = f"{path}.log"

    def log(self, text, datetime=False):
        if self.rotating:
            if os.stat(self.path).st_size < self.max_bytes:
                self.__write(text, self.path)
            else:
                pass
        else:
            self.__write(text, self.path)

    def __write(self, text, path):
        with open(path, 'a') as file_handler:
            file_handler.write(f"{text}\n")
