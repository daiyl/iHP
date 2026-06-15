import torch
from pathlib import Path

from phenotype_decoders.base_decoder import decode_state, load_decoder, parse_state, state_label

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_dir = Path("phenotype_decoders/checkpoints")
output_dir = Path("phenotype_decoders/observation_data/generated_observations")
output_dir.mkdir(parents=True, exist_ok=True)

def multi_decoder(state):
	decoders = {"pneumonia":"pneumonia_decoder.pt", "fundus":"fundus_decoder.pt", "oct":"oct_decoder.pt"}
	images = []
	for name, dec in decoders.items():
		decoder = load_decoder(dec, checkpoint_dir, device)
		image = decode_state(decoder, state, device, round_to_training_state=False)
		images.append(image)
		# plt.figure()
		# plt.imshow(image)
		# output_path = output_dir / f"{name}-{sn}.png"
		# image.save(output_path)
		# print(f"Saved {output_path}")
		
	return images