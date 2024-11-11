import json
import requests
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from sklearn.metrics import classification_report as sk_classification_report
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
import random
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

# Entities and relations
entities_list = ['cpf', 'nome', 'endereco', 'rg']
relations = ['mora em', 'associado']

# Create label list for NER
label_list = ['O']
for entity in entities_list:
    label_list.extend(['B-' + entity, 'I-' + entity])
label_list = sorted(set(label_list))
label_map = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label_map.items()}

# Relation labels
relation_labels = relations  # Exclude 'no_relation' from target labels
relation_label_map = {label: idx for idx, label in enumerate(relation_labels)}
id2relation = {idx: label for label, idx in relation_label_map.items()}


def load_dataset(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_line = json.loads(line)
            data.append(json_line)
    return data


def extract_annotations(data):
    dataset = []
    for item in data:
        # Fetch the text content
        text_url = item['data_row']['row_data']
        text_response = requests.get(text_url)
        text_response.encoding = 'utf-8'  # Ensure the encoding is set to 'utf-8'
        text = text_response.text.lower()  # Convert text to lowercase

        # Initialize entity and relation lists
        entities = []
        relations_list = []

        # Extract entities
        labels = item['projects'][list(item['projects'].keys())[0]]['labels']
        for label in labels:
            annotations = label['annotations']
            feature_id_to_entity = {}
            for obj in annotations.get('objects', []):
                entity_label = obj['value'].lower()
                if entity_label in entities_list:
                    entity = {
                        'start': obj['location']['start'],
                        'end': obj['location']['end'],
                        'label': entity_label,
                        'feature_id': obj['feature_id']
                    }
                    entities.append(entity)
                    feature_id_to_entity[obj['feature_id']] = entity

            # Extract relations
            for rel in annotations.get('relationships', []):
                relation_name = rel['name']
                if relation_name in relations:
                    source_id = rel['unidirectional_relationship']['source']
                    target_id = rel['unidirectional_relationship']['target']
                    if (source_id in feature_id_to_entity and
                            target_id in feature_id_to_entity):
                        relation = {
                            'type': relation_name,
                            'head': feature_id_to_entity[source_id],
                            'tail': feature_id_to_entity[target_id]
                        }
                        relations_list.append(relation)

        dataset.append({
            'text': text,
            'entities': entities,
            'relations': relations_list
        })
    return dataset


def convert_to_bio_format(dataset, tokenizer):
    tokenized_dataset = []
    for data in dataset:
        text = data['text']
        entities = data['entities']
        relations = data['relations']

        # Tokenize and get offset mappings
        tokenized_input = tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            padding='max_length',
            max_length=512  # Adjust max_length as needed
        )
        tokens = tokenizer.convert_ids_to_tokens(tokenized_input['input_ids'])
        offset_mapping = tokenized_input['offset_mapping']

        labels = ['O'] * len(tokens)

        # Assign labels to tokens
        for entity in entities:
            entity_start = entity['start']
            entity_end = entity['end']
            entity_label = entity['label']
            for idx, (start, end) in enumerate(offset_mapping):
                if start is None or end is None:
                    continue  # Skip special tokens
                if start >= entity_end:
                    break
                if end <= entity_start:
                    continue
                if start >= entity_start and end <= entity_end:
                    if labels[idx] == 'O':
                        labels[idx] = 'B-' + entity_label
                    elif labels[idx].startswith('B-') or labels[idx].startswith('I-'):
                        labels[idx] = 'I-' + entity_label

        # Truncate or pad labels to max_length
        labels = labels[:512]
        labels += ['O'] * (512 - len(labels))

        # Prepare relation data (entity pairs)
        tokenized_dataset.append({
            'input_ids': tokenized_input['input_ids'],
            'attention_mask': tokenized_input['attention_mask'],
            'labels': labels,
            'entities': entities,
            'relations': relations,
            'offset_mapping': offset_mapping,
        })
    return tokenized_dataset


def prepare_dataset(tokenized_dataset, label_map):
    features = []
    for item in tokenized_dataset:
        labels = [label_map.get(label, label_map['O']) for label in item['labels']]
        features.append({
            'input_ids': item['input_ids'],
            'attention_mask': item['attention_mask'],
            'labels': labels,
            'entities': item['entities'],
            'relations': item['relations'],
            'offset_mapping': item['offset_mapping'],
        })
    return Dataset.from_list(features)


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    report = classification_report(
        true_labels, true_predictions, output_dict=True, zero_division=0
    )

    return {
        'eval_precision': precision_score(true_labels, true_predictions),
        'eval_recall': recall_score(true_labels, true_predictions),
        'eval_f1': f1_score(true_labels, true_predictions),
        'eval_report': report
    }


def train_and_evaluate(train_dataset, test_dataset, tokenizer):
    model = AutoModelForTokenClassification.from_pretrained(
        'neuralmind/bert-base-portuguese-cased',
        num_labels=len(label_map),
        id2label=id2label,
        label2id=label_map
    ).to(device)  # Move model to GPU if available

    # Use DataCollatorForTokenClassification
    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir='./results',
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="no",
        save_strategy="no",
        num_train_epochs=3,  # Adjust as needed
        logging_steps=50,
        learning_rate=5e-5,
        weight_decay=0.01,
        disable_tqdm=True,  # Disable tqdm progress bars
        logging_dir='./logs',
        log_level='error',  # Set logging level to error
        # Removed 'use_cpu' parameter
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_result = trainer.evaluate()
    return eval_result, trainer  # Return evaluation results and trainer instance


def prepare_relation_dataset(dataset):
    relation_features = []
    for item in dataset:
        input_ids = item['input_ids']
        attention_mask = item['attention_mask']
        entities = item['entities']
        relations = item['relations']
        offset_mapping = item['offset_mapping']

        # Map entity positions to token indices
        entity_spans = []
        for entity in entities:
            entity_start = entity['start']
            entity_end = entity['end']
            start_idx = None
            end_idx = None
            for idx, (start, end) in enumerate(offset_mapping):
                if start is None or end is None:
                    continue
                if start <= entity_start < end:
                    start_idx = idx
                if start < entity_end <= end:
                    end_idx = idx
                if start_idx is not None and end_idx is not None:
                    break
            if start_idx is not None and end_idx is not None:
                entity_spans.append({
                    'start': start_idx,
                    'end': end_idx,
                    'label': entity['label'],
                    'feature_id': entity['feature_id']
                })

        # Generate all possible entity pairs
        for i, head in enumerate(entity_spans):
            for j, tail in enumerate(entity_spans):
                if i == j:
                    continue
                relation_label = 'no_relation'
                # Check if this pair exists in relations
                for rel in relations:
                    if (rel['head']['feature_id'] == head['feature_id'] and
                            rel['tail']['feature_id'] == tail['feature_id']):
                        relation_label = rel['type']
                        break
                relation_features.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'head_start': head['start'],
                    'head_end': head['end'],
                    'tail_start': tail['start'],
                    'tail_end': tail['end'],
                    'labels': relation_label_map.get(relation_label, -1),  # -1 for no_relation
                    'relation_label': relation_label  # Keep track for metrics
                })
    return relation_features


def prepare_relation_features(relation_features):
    features = []
    for item in relation_features:
        # Exclude instances with 'no_relation' labels
        if item['labels'] == -1:
            continue
        features.append({
            'input_ids': item['input_ids'],
            'attention_mask': item['attention_mask'],
            'labels': item['labels'],
            # Optionally include entity positions if needed
            'head_start': item['head_start'],
            'head_end': item['head_end'],
            'tail_start': item['tail_start'],
            'tail_end': item['tail_end'],
            'relation_label': item['relation_label'],  # Include for metrics
        })
    return Dataset.from_list(features)


def compute_relation_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids

    # Map numeric labels to relation names
    true_relations = [id2relation[label] for label in labels]
    pred_relations = [id2relation[pred] for pred in preds]

    report = sk_classification_report(
        labels, preds, target_names=relation_labels,
        output_dict=True, zero_division=0
    )
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']
    return {
        'eval_precision': precision,
        'eval_recall': recall,
        'eval_f1': f1,
        'eval_report': report
    }


# Custom Trainer subclass for relation extraction
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").to(model.device)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Adjust class weights based on your dataset's class distribution
        # You should calculate these counts based on your data
        class_counts = torch.tensor([2153, 2584], device=model.device)  # Prod pre full count
        total_counts = torch.sum(class_counts)
        class_weights = total_counts / class_counts

        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, model.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def train_relation_extraction(train_dataset, test_dataset, tokenizer):
    model = AutoModelForSequenceClassification.from_pretrained(
        'neuralmind/bert-base-portuguese-cased',
        num_labels=len(relation_label_map),
        id2label=id2relation,
        label2id=relation_label_map
    ).to(device)  # Move model to GPU if available

    # Use default data collator (it handles padding for sequence classification)
    data_collator = None

    training_args = TrainingArguments(
        output_dir='./relation_results',
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="no",
        save_strategy="no",
        num_train_epochs=3,  # Adjust as needed
        logging_steps=50,
        learning_rate=5e-5,
        weight_decay=0.01,
        disable_tqdm=True,  # Disable tqdm progress bars
        logging_dir='./relation_logs',
        log_level='error',  # Set logging level to error
        # Removed 'use_cpu' parameter
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_relation_metrics,
    )

    trainer.train()
    eval_result = trainer.evaluate()
    return eval_result


def split_dataset_random(dataset, test_size=0.2):
    # Create indices and shuffle them
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split_idx = int(len(dataset) * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    train_dataset = dataset.select(train_indices)
    test_dataset = dataset.select(test_indices)
    return train_dataset, test_dataset


def print_entity_examples(test_dataset, predictions, tokenizer, id2label, fold):
    """
    Function to print examples of identified entities and their detected values for each fold.
    """
    for i, test_example in enumerate(test_dataset):
        input_ids = test_example['input_ids']
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        # Convert predictions to entity labels
        pred_labels = np.argmax(predictions[i], axis=1)
        pred_labels = [id2label[pred] if pred != -100 else 'O' for pred in pred_labels]

        detected_entities = []
        current_entity = None
        current_entity_value = []

        # Loop through tokens and predictions to detect and accumulate entities
        for token, pred_label in zip(tokens, pred_labels):
            if pred_label.startswith("B-"):
                # If starting a new entity, save the previous one (if any)
                if current_entity:
                    detected_entities.append((current_entity, " ".join(current_entity_value)))
                current_entity = pred_label[2:]  # Remove "B-" to get the entity type
                current_entity_value = [token]  # Start accumulating tokens for the new entity
            elif pred_label.startswith("I-") and current_entity == pred_label[2:]:
                # Continue accumulating tokens for the current entity
                current_entity_value.append(token)
            else:
                # If the token is not part of a continuing entity, reset the current entity
                if current_entity:
                    detected_entities.append((current_entity, " ".join(current_entity_value)))
                current_entity = None
                current_entity_value = []

        # Handle last accumulated entity (if any)
        if current_entity:
            detected_entities.append((current_entity, " ".join(current_entity_value)))

        # Print the example and detected entities
        print(f"\nFold {fold + 1} - Example {i + 1}")
        print("Tokens:", tokens[:20])  # Limit to first 20 tokens for readability
        print("Detected Entities:")
        for entity, value in detected_entities:
            print(f"{entity.capitalize()} Detectado: {value}")

        # Optionally, break after a few examples to limit output
        if i >= 4:
            break  # Print only first 5 examples


# Main execution for training and printing examples (same as before)
if __name__ == "__main__":
    # Verify PyTorch can detect the GPU
    print("PyTorch version:", torch.__version__)
    print("Is CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU device name:", torch.cuda.get_device_name(0))
    else:
        print("No GPU detected.")

    file_path = 'corpus.ndjson'  # Replace with your dataset file path
    data = load_dataset(file_path)
    dataset = extract_annotations(data)

    tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')

    # Prepare the dataset
    tokenized_dataset = convert_to_bio_format(dataset, tokenizer)
    prepared_dataset = prepare_dataset(tokenized_dataset, label_map)

    # Prepare NER dataset (exclude non-tensor fields)
    ner_dataset = prepared_dataset.map(
        lambda example: {
            'input_ids': example['input_ids'],
            'attention_mask': example['attention_mask'],
            'labels': example['labels'],
        }
    )

    # Prepare relation dataset
    relation_features = prepare_relation_dataset(prepared_dataset)
    relation_dataset = prepare_relation_features(relation_features)

    # Initialize lists to store metrics
    ner_precisions = []
    ner_recalls = []
    ner_f1s = []
    ner_entity_metrics = {entity: {'precision': [], 'recall': [], 'f1': []} for entity in entities_list}

    relation_precisions = []
    relation_recalls = []
    relation_f1s = []
    relation_metrics = {relation: {'precision': [], 'recall': [], 'f1': []} for relation in relations}

    num_folds = 10  # Number of folds for cross-validation

    for fold in range(num_folds):
        print(f"Starting fold {fold + 1}/{num_folds}...")

        # Split the datasets randomly
        train_dataset, test_dataset = split_dataset_random(ner_dataset, test_size=0.2)  # 80/20 split
        train_relation_dataset, test_relation_dataset = split_dataset_random(relation_dataset, test_size=0.2)  # 80/20 split

        # Train and evaluate NER model
        ner_results, trainer = train_and_evaluate(train_dataset, test_dataset, tokenizer)
        ner_precisions.append(ner_results['eval_precision'])
        ner_recalls.append(ner_results['eval_recall'])
        ner_f1s.append(ner_results['eval_f1'])

        # Collect per-entity metrics
        report = ner_results['eval_report']
        for entity in entities_list:
            if entity in report:
                ner_entity_metrics[entity]['precision'].append(report[entity]['precision'])
                ner_entity_metrics[entity]['recall'].append(report[entity]['recall'])
                ner_entity_metrics[entity]['f1'].append(report[entity]['f1-score'])

        # Get predictions from the model
        predictions, labels, _ = trainer.predict(test_dataset)
        print_entity_examples(test_dataset, predictions, tokenizer, id2label, fold)

        # Train and evaluate relation extraction model
        relation_results = train_relation_extraction(
            train_relation_dataset, test_relation_dataset, tokenizer
        )
        # Corrected keys
        relation_precisions.append(relation_results['eval_precision'])
        relation_recalls.append(relation_results['eval_recall'])
        relation_f1s.append(relation_results['eval_f1'])

        # Collect per-relation metrics
        report = relation_results['eval_report']
        for relation in relations:
            if relation in report:
                relation_metrics[relation]['precision'].append(report[relation]['precision'])
                relation_metrics[relation]['recall'].append(report[relation]['recall'])
                relation_metrics[relation]['f1'].append(report[relation]['f1-score'])

    # Calculate average metrics
    avg_ner_precision = np.mean(ner_precisions)
    avg_ner_recall = np.mean(ner_recalls)
    avg_ner_f1 = np.mean(ner_f1s)

    avg_relation_precision = np.mean(relation_precisions)
    avg_relation_recall = np.mean(relation_recalls)
    avg_relation_f1 = np.mean(relation_f1s)

    # Print averaged NER metrics
    print("\nAverage NER Metrics over 10 folds:")
    print(f"Precision: {avg_ner_precision:.2f}")
    print(f"Recall: {avg_ner_recall:.2f}")
    print(f"F1-score: {avg_ner_f1:.2f}\n")

    # Print average per-entity metrics
    print("Average NER Metrics per Entity over 10 folds:")
    for entity in entities_list:
        if len(ner_entity_metrics[entity]['precision']) > 0:
            avg_precision = np.mean(ner_entity_metrics[entity]['precision'])
            avg_recall = np.mean(ner_entity_metrics[entity]['recall'])
            avg_f1 = np.mean(ner_entity_metrics[entity]['f1'])
            print(f"Entity: {entity}")
            print(f"  Precision: {avg_precision:.2f}")
            print(f"  Recall: {avg_recall:.2f}")
            print(f"  F1-score: {avg_f1:.2f}\n")

    # Print averaged Relation Extraction metrics
    print("Average Relation Extraction Metrics over 10 folds:")
    print(f"Precision: {avg_relation_precision:.2f}")
    print(f"Recall: {avg_relation_recall:.2f}")
    print(f"F1-score: {avg_relation_f1:.2f}\n")

    # Print average per-relation metrics
    print("Average Relation Extraction Metrics per Relation over 10 folds:")
    for relation in relations:
        if len(relation_metrics[relation]['precision']) > 0:
            avg_precision = np.mean(relation_metrics[relation]['precision'])
            avg_recall = np.mean(relation_metrics[relation]['recall'])
            avg_f1 = np.mean(relation_metrics[relation]['f1'])
            print(f"Relation: {relation}")
            print(f"  Precision: {avg_precision:.2f}")
            print(f"  Recall: {avg_recall:.2f}")
            print(f"  F1-score: {avg_f1:.2f}\n")