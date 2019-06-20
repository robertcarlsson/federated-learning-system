from server.federation import Federation

if __name__ == '__main__':
    from federation import Federation

class Server:
    def __init__(self):
        pass
    
    def create_federation(self):
        self.federation = Federation()

    def print_federation(self):
        print(self.federation)

    def train_federation(self, n_rounds):
        self.federation.train_federation_epoch(n_rounds)
    
    def get_results(self):
        return str(self.federation.results)
        