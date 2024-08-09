class IdGiver:
    def __init__(self):
        self.id = 0

    def giveId(self):
        idToRet = self.id
        self.id += 1
        return idToRet