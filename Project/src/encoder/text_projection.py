from torch import nn
import torch
from torch.nn.utils.rnn import pad_sequence
import omegaconf


class TextProjection(nn.Module):
    """ Class that projects raw text to g^3 dimensional vector by transforming
        the raw text to an embeddding using bert then applying an MLP
    """

    def __init__(self, opt, grid_size=8):
        super().__init__()
        self.grid_size = grid_size
        # Loading pre-trained tokenizer and bert model [can freeze here for faster training]

        self.tokenizer = torch.hub.load(
            'huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')
        self.bert = torch.hub.load('huggingface/pytorch-transformers',
                                   'model', 'bert-base-uncased')

        configs = omegaconf.OmegaConf.load(opt.config_path)
        model_params = configs.model.params
        # Number of dimensions output by the bert model
        self.BERT_OUTPUT_DIMENSTIONS = 768
        self.device = opt.device
        self.projection = nn.Sequential()
        # Initialize projection model
        self.create_projection_module(model_params)

        self.to(opt.device)

    def forward(self, input):
        """Forward process

        Args:
            input  string[batch_size]: 1D array containing all texts in batch 

        Returns:
            projection FloatTensor[batch_size, self.grid_size,self.grid_size,self.grid_size] : 4D tensor containing all projections in a single batch
        """
        tokenizer = self.tokenizer(input)
        raw_tokens = tokenizer.get('input_ids')
        batch, mask = self.pad_tokens(raw_tokens)
        bert_embeddings = self.bert(batch, attention_mask=mask)
        bert_embeddings = bert_embeddings.last_hidden_state
        # From the paper => "The first in the last layer of the BERTBASE model as the text feature embedding "
        bert_embeddings = bert_embeddings[:, 0, :]
        batch_size = bert_embeddings.shape[0]
        projection = self.projection(bert_embeddings)
        projection = projection.reshape(
            (batch_size, self.grid_size, self.grid_size, self.grid_size))

        return projection

    def pad_tokens(self, tokens):
        """Converts variable length token list to have equal lengths so that it can be processed by bert

        Args:
            tokens int[batch_size,n]: tokens of any length output by the tokenizer

        Returns:
            batch LongTensor[batch_size, max_sequence_length] : Long tensor of equal lengths padded by the tokenizer pad_token
            mask BoolTensor[batch_size, max_sequence_length]  : Tensor that indicates padding positions 0 => True and 1 => False
        """
        tensor_tokens = [torch.LongTensor(token) for token in tokens]
        padded_sequnce = pad_sequence(
            tensor_tokens, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        batch = padded_sequnce.to(self.device)
        mask = torch.BoolTensor(padded_sequnce.shape).to(self.device)
        mask[:, :] = True
        mask[padded_sequnce == self.tokenizer.pad_token_id] = False
        return batch, mask

    def create_projection_module(self, configs):
        n_layers = configs.get("n_layers")
        hidden_dim = configs.get("hidden_dim")
        apply_batch_norm = configs.get("apply_batch_norm")
        previous_hidden_dim = self.BERT_OUTPUT_DIMENSTIONS
        for i in range(n_layers):
            current_hidden_dim = hidden_dim[i] if isinstance(
                hidden_dim, list) else hidden_dim
            if (i == n_layers-1):
                current_hidden_dim = self.grid_size**3
            self.projection.append(
                nn.Linear(previous_hidden_dim, current_hidden_dim))
            if (apply_batch_norm):
                self.projection.append(nn.BatchNorm1d(current_hidden_dim))
            self.projection.append(nn.ReLU())
            previous_hidden_dim = current_hidden_dim


# Demo
# vq_cfg = 'configs/text_projection_configs.yaml'
# options = BaseOptions(config_path=vq_cfg)

# textProjection = TextProjection(options)
# x = ["Who was Jim Henson ?", "Jim Henson was a puppeteer", "hi tehere"]
# textProjection.forward(x)
