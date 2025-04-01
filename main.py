from generator import Generator

def main():
    generator = Generator()
    query = "What is machine learning?"
    response = generator.chat(query)
    print(response)

if __name__ == "__main__":
    main()
