from server.federation import Federation

class Server:
    def __init__(self):
        pass
    
    def create_federation(self):
        self.federation = Federation()

    def print_federation(self):
        print(self.federation)
        