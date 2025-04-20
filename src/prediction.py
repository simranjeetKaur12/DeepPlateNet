import matplotlib.pyplot as plt
import cv2
import numpy as np

model.eval()
with torch.no_grad():
    for batch in train_loader:
        images, labels, label_lengths = batch  
        images = images.to("cuda")  

     
        max_len = max(len(l) for l in labels)
        padded_labels = torch.zeros(len(labels), max_len, dtype=torch.long)
        for i, label in enumerate(labels):
            padded_labels[i, :len(label)] = label

        output = model(images) 
        output = output.log_softmax(2)  

        
        def decode_prediction(output, idx_to_char):
            pred = output.argmax(2).permute(1, 0)  
            texts = []
            for p in pred:
                chars = []
                prev = None
                for idx in p:
                    idx = idx.item()
                    if idx != prev and idx != 0:  
                        chars.append(idx_to_char.get(idx, ''))
                    prev = idx
                texts.append(''.join(chars))
            return texts

        decoded_texts = decode_prediction(output.cpu(), idx_to_char)
        print("Predicted:", decoded_texts)
        

def show_predictions(dataset, model, idx_to_char, num_samples=8):
    model.eval()
    fig, axs = plt.subplots(1, num_samples, figsize=(18, 4))
    with torch.no_grad():
        for i in range(num_samples):
            img, label_tensor, _ = dataset[i]
            img_input = img.unsqueeze(0).to("cuda")  # Add batch dim
            output = model(img_input)
            pred_text = decode_prediction(output.cpu(), idx_to_char)[0]

            # Convert image tensor to displayable image
            img_np = img.squeeze().numpy()  # [H, W]
            img_np = (img_np * 0.5 + 0.5) * 255  # unnormalize
            img_np = img_np.astype(np.uint8)

            axs[i].imshow(img_np, cmap='gray')
            axs[i].set_title(f'Pred: {pred_text}', fontsize=10)
            axs[i].axis('off')
    plt.tight_layout()
    plt.show()

show_predictions(dataset, model, idx_to_char)