from said.model.diffusion import SAID

if __name__ == "__main__":
    said_model = SAID()
    print(said_model.forward("forward is working"))
