# Attention Is All You Need: English to Italian Translation

This repository contains the code for replicating the original "Attention is All You Need" paper by Vaswani et al., for the task of text translation from English to Italian. The code is structured to closely follow the architecture and training process outlined in the paper, employing the Transformer model to achieve state-of-the-art performance on the translation task.

![Transformer Model Image](https://github.com/darshanvjani/ERA_vision_nlp_ai/assets/35656144/7c2f9a1e-1cad-4461-a182-e6943a35a484)

## 📄 Original Paper

The original paper by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin introduced the Transformer model, which solely relies on self-attention mechanisms without relying on recurrence. The paper proposed a new simple network architecture, the Transformer, based on self-attention, dispensing with recurrence and convolutions entirely.

[Read the Original Paper](https://arxiv.org/abs/1706.03762)

## 💡 Project Highlights

1. **Replication of Original Paper**: The code in this repository aims to closely replicate the findings and architecture proposed in the original paper.
2. **English to Italian Translation**: Customized the Transformer model for the specific task of translating English text to Italian.
3. **Code Structure**: The code is organized to reflect the structure of the Transformer model as outlined in the original paper, facilitating a deeper understanding of the model and its components.

## 📚 Data Exploration

The dataset utilized for this project involves translation pairs from English to Italian. Here are some examples from the dataset:

```plaintext
SOURCE: The quick brown fox jumps over the lazy dog.
TARGET: La rapida volpe marrone salta sopra il cane pigro.

SOURCE: Hello World!
TARGET: Ciao Mondo!

SOURCE: Attention is the key to unlocking the full potential of a Transformer network.
TARGET: L'attenzione è la chiave per sbloccare il pieno potenziale di una rete Transformer.


## Model Training

Final Train Loss: 1.872 after training it for 30 epochs.

## Training Logs
```--------------------------------------------------------------------------------
SOURCE: And yet at that very point of her recollections when she remembered Vronsky, the feeling of shame grew stronger and some inner voice seemed to say to her, 'warm, very warm, burning!' 'Well, what of it?' she finally said to herself with decision, changing her position on the seat.
TARGET: Ma intanto proprio a questo punto il senso di vergogna diveniva più forte, come se una certa voce interiore, proprio lì, nel punto in cui si ricordava di Vronskij, le dicesse: «Caldo, caldo, scottante».
PREDICTED: E pure in questo punto , nel ricordo dei ricordi in cui ella ricordò di lei , il sentimento di vergogna di vergogna e di voce , di lui , le sembrava avere addosso , di nuovo una mano . — E di nuovo , cosa ? — disse , alla stessa cosa in cui era scritto tutto . — E cambiò una domanda ? — E cambiò posizione ?
--------------------------------------------------------------------------------
SOURCE: "What did Tom say about those cheeses?"
TARGET: — Che ha detto Tommaso di quel formaggio?
PREDICTED: — Che Tommaso Smith ? — chiese tutti i due versi ?
--------------------------------------------------------------------------------
SOURCE: "After all, a single morning's interruption will not matter much," said he, "when I mean shortly to claim you--your thoughts, conversation, and company--for life."
TARGET: — Del resto, che cosa importa d'imporsi un po' di ritegno per una mattinata? Fra poco avrò i vostri pensieri, la vostra compagnia e voi tutta per la vita intera.
PREDICTED: — Ma , una mattina di attesa — egli non si è molto probabile che quando ve lo dica : " quando vi un conto e la vostra vita .
--------------------------------------------------------------------------------
SOURCE: We know what sort of soil his is, black as poppy-seed, but he cannot boast of his harvests either.
TARGET: Noi sappiamo che la terra è la sua: una bellezza; pure anche lui non ha a lodarsi del raccolto.
PREDICTED: Noi sappiamo che cosa ha un ricca di carne , ma che non può la schiena del suo ritorno , o non può dei suoi consigli .
```
