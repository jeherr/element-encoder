import element_coder as ecoder

def main():
	network = ecoder.ElementCoder(latent_size=4, batches_per_epoch=100, hidden_layers=[128, 128],
				max_steps=2000, batch_size=64, learning_rate=0.0001, test_freq=5)
	network.train()

if __name__ == "__main__":
	main()
