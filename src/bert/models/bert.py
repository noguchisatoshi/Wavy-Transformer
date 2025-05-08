import torch
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification
from models.embeddings import CustomBertEmbeddings
from models.layers import CustomBertLayer 
from collections import OrderedDict

class CustomBertEncoder(nn.Module):
    def __init__(self,
                 num_hidden_layers=12,
                 hidden_size=768,
                 intermediate_size=3072,
                 num_attention_heads=12,
                 dropout_prob=0.1,
                 residual_type="diffuse",   # "diffuse", "wave", "mix" のいずれか
                 tau=1.0,
                 ):
        super().__init__()
        
        self.residual_type = residual_type
        
        self.layer = nn.ModuleList([
            CustomBertLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                dropout_prob=dropout_prob,
                residual_type=residual_type,
                tau=tau,
                apply_velocity_transform=(i != num_hidden_layers - 1),  # 最終層ならFalse
            ) for i in range(num_hidden_layers)
        ])


    def forward(self, hidden_states, attention_mask=None,
                output_hidden_states=True, output_attentions=False):
                
        all_hidden_states = []
        all_attentions = []

        if self.residual_type in ["wave", "mix"]:
            previous_hidden_velocity = torch.zeros_like(hidden_states)


        for i, layer_module in enumerate(self.layer): 
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
                
            if self.residual_type == "diffuse":
                hidden_states, attn_weights = layer_module(hidden_states, attention_mask=attention_mask)
            
            elif self.residual_type in ["wave", "mix"]:
                hidden_states, velocity, attn_weights = layer_module(
                    hidden_states, 
                    attention_mask=attention_mask, 
                    previous_hidden_velocity=previous_hidden_velocity
                )
                previous_hidden_velocity = velocity
    
            else:
                raise ValueError(f"Unknown residual_type: {self.residual_type}")
            
            if output_attentions:
                all_attentions.append(attn_weights)
        
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs += (all_hidden_states,)
        if output_attentions:
            outputs += (all_attentions,)

        return outputs

class CustomBertModel(nn.Module):
    def __init__(self,
                 config: BertConfig,
                 residual_type="diffuse",
                 tau=1.0,
                 ):
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
        
        encoder_outputs = self.encoder(
            embeddings,
            attention_mask=extended_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
        )
        
        last_hidden_state = encoder_outputs[0]
        pooled_output = self.pooler(last_hidden_state[:, 0, :])

        outputs = (last_hidden_state, pooled_output) + encoder_outputs[1:]
        return outputs

class CustomBertForSequenceClassification(nn.Module):
    def __init__(self, config: BertConfig, residual_type="diffuse", tau=1.0):
        super().__init__()
        self.num_labels = config.num_labels
        
        self.bert = CustomBertModel(config, residual_type=residual_type,
                                    tau=tau,)
        
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None,
                output_hidden_states=True, output_attentions=False):
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
        )
        
        pooled_output = outputs[1]

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1).float())
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return (loss, logits) + outputs[2:]

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, residual_type="diffuse", num_labels=2,):
        config = BertConfig.from_pretrained(pretrained_name_or_path)
        config.num_labels = num_labels

        model = cls(config, residual_type=residual_type,
                    use_sparse_attention=use_sparse_attention,
                    window_size=window_size)

        hf_model = BertForSequenceClassification.from_pretrained(
            pretrained_name_or_path,
            num_labels=num_labels
        )
        hf_state_dict = hf_model.state_dict()

        missing_keys, unexpected_keys = model.load_state_dict(hf_state_dict, strict=False)

        if missing_keys:
            print("Missing keys (not found in pretrained):", missing_keys)
        if unexpected_keys:
            print("Unexpected keys (unused):", unexpected_keys)

        return model

class CustomBertForQuestionAnswering(nn.Module):
    def __init__(self, config: BertConfig, residual_type="diffuse", tau=1.0):
        super().__init__()
        self.bert = CustomBertModel(config, residual_type=residual_type,
                                    tau=tau,)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                start_positions=None, end_positions=None,
                output_hidden_states=True, output_attentions=False):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            output_hidden_states=output_hidden_states,
                            output_attentions=output_attentions)
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)  # [batch_size, seq_length, 2]
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # [batch_size, seq_length]
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return (total_loss, start_logits, end_logits) + outputs[2:]