import element_coder as ecoder

def main():
	network = ecoder.ElementCoder(latent_size=4, batches_per_epoch=100)
	network.train()

if __name__ == "__main__":
	main()
