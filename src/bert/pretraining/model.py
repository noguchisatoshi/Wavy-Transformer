# models/model.py
import torch
import torch.nn as nn
from transformers import BertConfig
from collections import OrderedDict
from models.embeddings import CustomBertEmbeddings
from models.layers import CustomBertLayer

class CustomBertEncoder(nn.Module):
    def __init__(self, num_hidden_layers=12, hidden_size=768, intermediate_size=3072,
                 num_attention_heads=12, dropout_prob=0.1, residual_type="diffuse", tau=1.0):
        super().__init__()
        self.residual_type = residual_type
        self.add_mass = add_mass
        self.layer = nn.ModuleList([
            CustomBertLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                dropout_prob=dropout_prob,
                residual_type=residual_type,
                tau=tau,
                apply_velocity_transform=(i != num_hidden_layers - 1),
            ) for i in range(num_hidden_layers)
        ])

    def forward(self, hidden_states, attention_mask=None):
        all_hidden_states = []
        all_attentions = []

        if self.residual_type in ["wave", "mix"]:
            previous_hidden_velocity = torch.zeros_like(hidden_states)
            
        for i, layer in enumerate(self.layer):
            all_hidden_states.append(hidden_states)
            if self.residual_type == "diffuse":
                hidden_states, attn_weights = layer(hidden_states, attention_mask=attention_mask)
            
            elif self.residual_type in ["wave", "mix"]:
                hidden_states, velocity, attn_weights = layer(
                    hidden_states, 
                    attention_mask=attention_mask, 
                    previous_hidden_velocity=previous_hidden_velocity
                )
                previous_hidden_velocity = velocity

            else:
                raise ValueError(f"Unknown residual_type: {self.residual_type}")
            all_attentions.append(attn_weights)

        all_hidden_states.append(hidden_states)
        return hidden_states, all_hidden_states, all_attentions

class CustomBertModel(nn.Module):
    def __init__(self, config: BertConfig, residual_type="diffuse", tau=1.0):
        super().__init__()
        self.config = config
        self.embeddings = CustomBertEmbeddings(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob
        )
        self.encoder = CustomBertEncoder(
            num_hidden_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            dropout_prob=config.hidden_dropout_prob,
            residual_type=residual_type,
            tau=tau,
        )
        self.pooler = nn.Sequential(OrderedDict([
            ('dense', nn.Linear(config.hidden_size, config.hidden_size)),
            ('activation', nn.Tanh())
        ]))
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                output_hidden_states=True, output_attentions=False):
        if attention_mask is not None:
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2).float()
            extended_mask = (1.0 - extended_mask) * -1e9
        else:
            extended_mask = None

        embeddings = self.embeddings(input_ids, token_type_ids)
        encoder_outputs = self.encoder(embeddings, attention_mask=extended_mask)
        last_hidden_state = encoder_outputs[0]
        pooled_output = self.pooler(last_hidden_state[:, 0, :])
        
        hidden_states_out = encoder_outputs[1] if output_hidden_states else None
        attentions_out = encoder_outputs[2] if output_attentions else None
        return (last_hidden_state, pooled_output, hidden_states_out, attentions_out)

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertLMPredictionHead(nn.Module):
    def __init__(self, config, embedding_weights):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.weight = embedding_weights
    
    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, embedding_weights):
        super().__init__()
        self.predictions = BertLMPredictionHead(config, embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)
    
    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score

class CustomBertForPreTraining(nn.Module):
    def __init__(self, config: BertConfig, residual_type="diffuse", tau=1.0):
        super().__init__()
        self.bert = CustomBertModel(
            config, 
            residual_type=residual_type,
            tau=tau, 
        )
        embedding_weights = self.bert.embeddings.word_embeddings.weight
        self.cls = BertPreTrainingHeads(config, embedding_weights)
        self.config = config

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                masked_lm_labels=None, next_sentence_label=None,
                output_hidden_states=False, output_attentions=False,
                compute_nsp_loss=True):
        outputs = self.bert(
            input_ids, 
            attention_mask, 
            token_type_ids,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
        )

        sequence_output, pooled_output, hidden_states, attentions = outputs
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        
        total_loss = None
        mlm_loss = None
        
        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
            mlm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1)
            )
            if next_sentence_label is not None and compute_nsp_loss:
                nsp_loss = loss_fct(
                    seq_relationship_score.view(-1, 2),
                    next_sentence_label.view(-1)
                )
                total_loss = mlm_loss + nsp_loss
            else:
                total_loss = mlm_loss

        if output_hidden_states or output_attentions:
            return total_loss, prediction_scores, seq_relationship_score, hidden_states, attentions
        else:
            return total_loss, prediction_scores, seq_relationship_score