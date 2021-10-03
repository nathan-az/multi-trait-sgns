import torch
from torch import nn
import pytest

from models.sgns import MultiEmbeddingSGNS


@pytest.fixture(scope="session")
def mock_training_data():
    # There are 2 types of embeddings and 2 batches.
    # For first batch, the target indices are (1, 1) and context (2, 2)
    X = torch.tensor(
        [
            [[1, 1], [2, 2]],
            [[0, 0], [3, 3]],
        ]
    )
    y = torch.tensor(
            [1., 1.]
    ).double()
    return X, y


# scoped as function so we can perform backprop without breaking tests
@pytest.fixture(scope="function")
def mock_model():
    model = MultiEmbeddingSGNS(vocab_size=5, embedding_dim=3, side_info_specs={"s1": 4})
    for i, embedding in enumerate(
        zip(model.target_embeddings.values(), model.context_embeddings.values())
    ):
        shape = embedding[0].weight.shape
        new_weights = torch.meshgrid([torch.arange(shape[0]), torch.arange(shape[1])])[
            0
        ].double() + float(5 * i)
        embedding[0].weight = nn.Parameter(new_weights)
        embedding[1].weight = nn.Parameter(-1 * new_weights)

    weights_shape = model.target_embedding_weights.shape
    model.target_embedding_weights = nn.Parameter(
        torch.ones(*weights_shape).double() * 2
    )
    model.context_embedding_weights = nn.Parameter(
        torch.ones(*weights_shape).double() * 2
    )
    return model


def test_indices_to_embedding_vectors(mock_model, mock_training_data):
    mock_indices = mock_training_data[0]
    expected = torch.tensor(
        [[[1.0, 1.0, 1.0], [6.0, 6.0, 6.0]], [[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]]]
    ).double()
    actual = mock_model.indices_to_embedding_vectors(
        mock_indices[:, 0, :],
        embeddings=[embedding for embedding in mock_model.target_embeddings.values()],
    )
    assert torch.equal(expected, actual)


def test_forward_target(mock_model, mock_training_data):
    mock_indices = mock_training_data[0]
    expected = torch.tensor([[[3.5, 3.5, 3.5]], [[2.5, 2.5, 2.5]]]).double()
    actual = mock_model.forward_target(
        mock_indices[:, 0, :],
    )
    assert torch.equal(expected, actual)


def test_forward_correct_shape(mock_model, mock_training_data):
    mock_indices = mock_training_data[0]
    actual = mock_model.forward(mock_indices)
    assert tuple(actual.shape) == (2,)

def test_backprop_works(mock_model, mock_training_data):
    original_base_target = torch.clone(mock_model.target_embeddings["base"].weight.data)
    optimizer = torch.optim.SGD(mock_model.parameters(), lr=0.2)
    loss_fn = nn.BCELoss()
    X, y = mock_training_data

    preds = mock_model(X)

    loss = loss_fn(preds, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    after_base_target = mock_model.target_embeddings["base"].weight.data

    assert not torch.equal(original_base_target, after_base_target)
    print("Woohoo!")
