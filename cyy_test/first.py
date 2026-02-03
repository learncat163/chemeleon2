from src.sample import sample

# Sample 1000 structures with Default model trained with alex-mp-20
gen_atoms_list = sample(
    num_samples=1000, 
    batch_size=500, 
    output_dir="outputs/alex-mp-20",
)

# Each structure is an ASE Atoms object
from ase.visualize import view

view(gen_atoms_list[0], viewer="ngl")