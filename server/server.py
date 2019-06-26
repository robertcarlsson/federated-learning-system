import subprocess

class Server:
    def __init__(self):
        pass
    
    def create_federation(self):
        print('Starting subprocess')
        subprocess.call('/app/create_federation.sh', shell=True)
        print('End of subprocess')
        pass

    def print_federation(self):
        pass

    def train_federation(self, n_rounds):
        pass
    
    def get_results(self):
        pass
        