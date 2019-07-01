import os
import docker
import subprocess


client = docker.from_env()

class Server:
    def __init__(self):
        pass
    
    def create_federation(self):
        print('Starting subprocess')
        subprocess.call('/app/create_federation.sh', shell=True)
        os.chdir('/src/federation')
        #os.system('docker build --rm=false -f "Dockerfile" -t federation:latest .')
        os.system('docker-compose up -d')
        #client.containers.run('federation_fed-srv')
        print(client.containers.list())
        
        print('End of subprocess')
        pass

    def print_federation(self):
        pass

    def train_federation(self, n_rounds):
        pass
    
    def get_results(self):
        pass
        