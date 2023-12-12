from collections.abc import Callable
from enum import auto

import torch
import numpy as np
from jaxtyping import Float
from strenum import LowercaseStrEnum
from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint


class Axis(LowercaseStrEnum):
    BATCH = auto()
    SEQUENCE = auto()
    MODEL_DIM = auto()
    
    @staticmethod
    def names(*axis: "Axis") -> str:
        """Join multiple axis together, to represent the dimensions of a tensor.

        Example:
            >>> print(Axis.names(Axis.BATCH, Axis.INPUT_OUTPUT_FEATURE))
            batch input_output_feature

        Args:
            *axis: Axis to join.

        Returns:
            Joined axis string.
        """
        return " ".join(a.value for a in axis)


MODEL_NAME = "EleutherAI/pythia-70m-deduped"
N_VECTORS = 20
LAYER_N = 3

def generate_with_hook(
    model: HookedTransformer,
    input: Float[Tensor, Axis.names(Axis.BATCH, Axis.SEQUENCE)],
    hook_point: str,
    steering_hook: Callable,
    length=50,
    temp=0.2,
):
    # TODO: kv cache
    for pos in range(length):
        # generate the output
        outputs = model.run_with_hooks(
            input, return_type="logits", fwd_hooks=[(hook_point, steering_hook)]
        )
        if temp == 0:
            # greedy decoding
            next_tokens = torch.argmax(outputs[:, -1, :], dim=-1)
        else:
            next_token_probabilities = torch.softmax(outputs[:, -1, :] , dim=-1)
            next_tokens = torch.multinomial(next_token_probabilities, num_samples=1)

        # append the last token to the input
        input = torch.cat([input, next_tokens], dim=-1)
    
    return input

def write_final_texts(tokens, tokenizer):
    for i in range(N_VECTORS):
        text = tokenizer.decode(tokens[i, :])
        print(f"Steering vector {i}: {text}")
        
def train_for_different_outputs(
    model: HookedTransformer,
    steering_vectors: Float[Tensor, Axis.names(Axis.BATCH, Axis.MODEL_DIM)],
    prompt_tokens: Float[Tensor, Axis.names(Axis.BATCH, Axis.SEQUENCE)],
    hook_point: str,
    steering_hook: Callable[[Tensor, HookPoint], Tensor | None],
    separation_function: Callable,
    n_iters: int = 100,
    length: int = 50,
    temperature: float = 0.2,
):
    optimizer = torch.optim.Adam([steering_vectors], lr=0.01)
    for iter_n in range(n_iters):
        # generate the output with the steering vectors
        input = prompt_tokens.repeat(N_VECTORS, 1)
        outputs = generate_with_hook(model, input, hook_point, steering_hook, length, temperature)

        # now we want to run the model one more time and get the mid_layer representation of the outputted tokens
        _, cache = model.run_with_cache(input, stop_at_layer=LAYER_N + 1)
        representations = cache[f"blocks.{LAYER_N}.hook_resid_post"][:, -1, :]

        separation_loss = separation_function(representations)
        optimizer.zero_grad()
        separation_loss.backward()
        optimizer.step()
        print(f"Separation loss: {separation_loss.item()}")
        
    write_final_texts(outputs)

    return representations

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model: HookedTransformer = HookedTransformer.from_pretrained(MODEL_NAME).to(device)
    tokenizer = model.tokenizer

    steering_dim = model.cfg.d_model

    # initialize the steering vectors
    steering_vectors = torch.randn(N_VECTORS, steering_dim, device=device) / np.sqrt(steering_dim)

    prompt = "The meaning of life is"
    tokenized_prompt = tokenizer(prompt, return_tensors="pt")

    # generate the output with the steering vectors
    prompt_tokens: torch.Tensor = tokenized_prompt["input_ids"].to(device)

    def steering_hook(value: Tensor, hook: HookPoint):
        # value is the output of the encoder
        # we can use this to steer the output
        # of the decoder
        return value + steering_vectors.unsqueeze(1).repeat(1, value.shape[1], 1)

    
    def separation_function(representations):
        repr_cos_sims = representations @ representations.T
        return (repr_cos_sims - torch.eye(repr_cos_sims.shape[0], device=device)).abs().mean()
    
    _ = train_for_different_outputs(
        model,
        steering_vectors,
        prompt_tokens,
        f"blocks.{LAYER_N}.hook_resid_post",
        steering_hook,
        separation_function,
    )

if __name__ == "__main__":
    main()