import pickle
import datasets
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
import torchvision.transforms as tv
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report


model_name = 'google/vit-base-patch16-224'

with open('dataset/data.pkl', 'rb') as picklefile:
    dataset = pickle.load(picklefile)
    print(dataset)
    print(dataset['train'].features)

    train_test_split = dataset['train'].train_test_split(test_size=0.1)
    train_val_split = train_test_split['train'].train_test_split(test_size=0.1)
    final_dataset = {
        'train': train_val_split['train'],
        'val': train_val_split['test'],
        'test': train_test_split['test']
    }

    train_ds = final_dataset['train']
    val_ds = final_dataset['val']
    test_ds = final_dataset['test']

    labels = train_ds.features['label'].names

    label2id = dict()
    id2label = dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    processor = ViTImageProcessor.from_pretrained(model_name)
    
    # Get Conf
    image_mean, image_std = processor.image_mean, processor.image_std
    size = processor.size['height']

    normalize = tv.Normalize(mean=image_mean, std=image_std)

    train_transforms = tv.Compose(
        [
            tv.RandomResizedCrop(size),
            tv.RandomHorizontalFlip(),
            tv.ToTensor(),
            normalize,
        ]
    )

    val_transforms = tv.Compose(
        [
            tv.Resize(size),
            tv.CenterCrop(size),
            tv.ToTensor(),
            normalize,
        ]
    )

    test_transforms = tv.Compose(
        [
            tv.Resize(size),
            tv.CenterCrop(size),
            tv.ToTensor(),
            normalize,
        ]
    )

    def apply_train_transforms(samples):
        samples['pixel_values'] = [train_transforms(image.convert('RGB')) for image in samples['image']]
        return samples

    def apply_test_transforms(samples):
        samples['pixel_values'] = [test_transforms(image.convert('RGB')) for image in samples['image']]
        return samples
    
    def apply_val_transforms(samples):
        samples['pixel_values'] = [val_transforms(image.convert('RGB')) for image in samples['image']]
        return samples
    
    train_ds.set_transform(apply_train_transforms)
    test_ds.set_transform(apply_test_transforms)
    val_ds.set_transform(apply_val_transforms)

    def collate_fn(samples):
        pixel_values = torch.stack([sample['pixel_values'] for sample in samples])
        labels = torch.tensor([sample['label']  for sample in samples])
        return {'pixel_values': pixel_values, 'labels': labels}
    
    train_dl = DataLoader(train_ds, collate_fn=collate_fn, batch_size=4)

    batch = next(iter(train_dl))
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(key, value.shape)
    
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    train_args = TrainingArguments(
        output_dir='models',
        per_device_train_batch_size=16,
        eval_strategy='steps',
        num_train_epochs=2,
        fp16=True,
        save_steps=10,
        eval_steps=10,
        logging_steps=10,
        learning_rate=2e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to='tensorboard',
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model,
        train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        tokenizer=processor,
    )

    trainer.train(resume_from_checkpoint=True)

    outputs = trainer.predict(test_ds)
    print(outputs.metrics)

    target_names = id2label.values()

    y_true = outputs.label_ids
    y_pred = outputs.predictions.argmax(1)

    print(classification_report(y_true, y_pred, target_names=target_names))