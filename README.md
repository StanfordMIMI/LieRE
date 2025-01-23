# LieRE: Generalizing Rotary Position Encodings

While Rotary Position Embeddings (RoPE) for large language models have become widely adopted, their application for other modalities has been slower. 
Here, we introduce Lie group Relative position Encodings (LieRE) that goes beyond RoPE in supporting n-dimensional inputs. We evaluate the performance of LieRE on 2D and 3D image classification tasks and observe that LieRE leads to marked relative improvements in performance (up to 9.7% for 2D and up to 25.5% for 3D), training efficiency (3.5x reduction), data efficiency (30%) compared to the baselines of DeiT III, RoPE-Mixed and Vision-Llama.

# Implementation for computing the rotation matrices
We here share the code for implementing the rotation matrices. In short, every rotation matrix can be represented as the matrix exponential of a skew-symmetric matrix and we make the matrix learnable by parametrizing the rotations with generators before the matrix exponential.
```python
generator_raw_params = nn.Parameter(
    torch.rand(
        input_dimensionality,
        head_dim,
        head_dim,
    ) * 2 * math.pi
)

upper_triangle = (
    torch.triu(generator_raw_params, diagonal=1)
)
skew_bases = upper_triangle - torch.transpose(upper_triangle, -1, -2)
in_basis_positions = (
    positions.reshape(list(positions.shape) + [1] * 2) * skew_bases
)
generator_pos = torch.sum(in_basis_positions, dim=-3)
rotation = torch.matrix_exp(generator_pos.to(dtype=torch.float32)).to(dtype=positions.dtype)
```
# Base repo
We used the transformer implementation and default hyperparameters of https://github.com/kentaroy47/vision-transformers-cifar10.

# Work in progress
To reproduce the results on CIFAR-100 use the follow command
```sbatch -c 48 --gres=gpu:l40:4 --nodelist=rae1 --time=00:00:00 lightning_cifar100.sh```
You can choose between the options `liere`, `rope_mixed`, `absolute` and `visionllama` for comparing position encodings. 

If you find this useful, please cite
```
@article{ostmeier2024liere,
  title={LieRE: Generalizing Rotary Position Encodings},
  author={Ostmeier, Sophie and Axelrod, Brian and Moseley, Michael E and Chaudhari, Akshay and Langlotz, Curtis},
  journal={arXiv preprint arXiv:2406.10322},
  year={2024}
}
```

# Other
Much of the code was branched from [vision-transformers-cifar10](https://github.com/kentaroy47/vision-transformers-cifar10).
