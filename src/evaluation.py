import torch 

def evaluate(model, dataloader, idx_to_char):
    model.eval()
    total_edit_distance = 0
    total_characters = 0
    correct_sequences = 0
    total_sequences = 0

    with torch.no_grad():
        for images, labels, label_lengths in dataloader:
            images = images.to("cuda")
            output = model(images)  # [W, B, num_classes]
            decoded_texts = decode_prediction(output.cpu(), idx_to_char)

            for pred, true_label, true_len in zip(decoded_texts, labels, label_lengths):
                true_text = ''.join([idx_to_char[i.item()] for i in true_label[:true_len]])

                dist = levenshtein_distance(pred, true_text)
                total_edit_distance += dist
                total_characters += len(true_text)

                if pred == true_text:
                    correct_sequences += 1
                total_sequences += 1

    char_accuracy = 1 - (total_edit_distance / total_characters)
    seq_accuracy = correct_sequences / total_sequences

    print(f"Character-level Accuracy: {char_accuracy:.4f}")
    print(f"Sequence-level Accuracy:  {seq_accuracy:.4f}")

evaluate(model, train_loader, idx_to_char)