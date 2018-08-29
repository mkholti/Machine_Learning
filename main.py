
from model import Model

def main():
    model = Model("fichierdeconf.json")
    model.train()
    model.score_report()

if __name__ == '__main__':
    main()
    