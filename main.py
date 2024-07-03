from settings import *
from train import *

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Utilizzo: python main.py -train OR python main.py -test [percorso_output]")
        return

    if len(sys.argv) >= 2:
        command = sys.argv[1]
        
    output_path = None
    if len(sys.argv) == 3:
        output_path = sys.argv[2]

    if command == "-train":
        print("Eseguo il training...")
        training_loop()
    elif command == "-test":
        print(f"Genero 500 immagini dal file di test ./test.txt con percorso di output: {output_path if output_path is not None else OUTPUT_PATH}")
        sample_500(MyDDPM())
    else:
        print("Argomento non valido. Utilizzo: python main.py -train OR python main.py -test [percorso_output]")

if __name__ == "__main__":
    main()
