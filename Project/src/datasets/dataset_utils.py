from torch.distributions import Categorical
from dataset_preprocessing.constants import GLOBAL_SEED
from utils import util
from torch.nn import functional
util.seed_everything(GLOBAL_SEED)


def sample_z_set(z_set):
    """ Transform z_set into a categorical distribution then sample

    Args:
        z_set (tensor[*]): tensor of any dimension represents a probability distribution
                           over the last dimension
    Returns:
        q_set (tensor[z_set.shape]): one hot encoded tensor with the same shape as z_set
    """

    # Transform z_set into a categorical distribution then sample
    # one hot encode for cross entropy loss  g^3 x 512
    num_classes = z_set.shape[-1]
    q_set = Categorical(z_set).sample()
    q_set = functional.one_hot(q_set, num_classes=num_classes)
    return q_set
