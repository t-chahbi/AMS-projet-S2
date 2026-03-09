# =============================================================================
# GÉNÉRATEUR DE TEXTE LSTM - Génère du texte caractère par caractère
# =============================================================================

import unidecode  # Convertit les accents en caractères simples (é -> e)
import string     # Contient tous les caractères imprimables
import random     # Pour choisir des morceaux aléatoires du texte
import re

from os import listdir, path, makedirs, popen
from os.path import isdir, isfile, join

import torch               # Bibliothèque de deep learning
import torch.nn as nn      # Contient les couches du réseau de neurones
from torch.autograd import Variable

import time, math

from argparse import ArgumentParser

# =============================================================================
# CONFIGURATION GPU/CPU
# On utilise le GPU si disponible car c'est 10-100x plus rapide
# =============================================================================
if torch.cuda.is_available():
	device = torch.device("cuda:0")  # GPU NVIDIA
	print('CUDA AVAILABLE')
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
	device = torch.device("mps")     # GPU Apple Silicon (M1, M2, M3)
	print('MPS (APPLE SILICON GPU) AVAILABLE')
else:
	device = torch.device("cpu")     # Pas de GPU, plus lent
	print('ONLY CPU AVAILABLE')

# =============================================================================
# CONSTANTES
# =============================================================================
all_characters = string.printable  # Tous les caractères possibles (a-z, A-Z, 0-9, ponctuation...)
n_characters = len(all_characters) # = 100 caractères différents
chunk_len = 100                    # Longueur des séquences pour l'entraînement

# Proportions pour diviser le corpus
TRAIN_RATIO = 0.8   # 80% pour apprendre
VAL_RATIO = 0.1     # 10% pour vérifier qu'on n'apprend pas par coeur
TEST_RATIO = 0.1    # 10% pour l'évaluation finale

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def time_since(since):
	"""Retourne le temps écoulé depuis 'since' au format 'Xm Ys'"""
	s = time.time() - since
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)

def split_corpus(text, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
	"""
	Divise le texte en 3 parties : train, validation, test
	
	Pourquoi ? Pour vérifier que le modèle généralise bien :
	- Train : le modèle apprend dessus
	- Validation : on vérifie qu'il n'apprend pas par coeur (overfitting)
	- Test : évaluation finale sur des données jamais vues
	"""
	total_len = len(text)
	train_end = int(total_len * train_ratio)
	val_end = int(total_len * (train_ratio + val_ratio))
	
	train_text = text[:train_end]
	val_text = text[train_end:val_end]
	test_text = text[val_end:]
	
	print(f'Corpus split:')
	print(f'  - Train: {len(train_text):,} chars ({train_ratio*100:.0f}%)')
	print(f'  - Validation: {len(val_text):,} chars ({val_ratio*100:.0f}%)')
	print(f'  - Test: {len(test_text):,} chars ({(1-train_ratio-val_ratio)*100:.0f}%)')
	
	return train_text, val_text, test_text

# =============================================================================
# CLASSE RNN (LSTM) - Le réseau de neurones
# =============================================================================
# Architecture :
#   Caractère -> [ENCODER] -> [LSTM] -> [DECODER] -> Caractère prédit
#
# LSTM = Long Short-Term Memory : garde une "mémoire" des caractères précédents
# =============================================================================

class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, n_layers=1):
		"""
		Crée le réseau de neurones LSTM
		
		Args:
			input_size: nombre de caractères possibles en entrée (100)
			hidden_size: taille de la mémoire interne (256)
			output_size: nombre de caractères possibles en sortie (100)
			n_layers: nombre de couches LSTM empilées (3)
		"""
		super(RNN, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers
		
		# ENCODER : transforme un caractère en vecteur de nombres
		# Ex: 'A' -> [0.2, -0.5, 0.8, ...] (256 nombres)
		self.encoder = nn.Embedding(input_size, hidden_size)
		
		# LSTM : traite la séquence et garde la mémoire du contexte
		self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers)
		
		# DECODER : transforme le vecteur en probabilités pour chaque caractère
		# Ex: [0.2, -0.5, ...] -> [P(a)=0.01, P(b)=0.05, P(c)=0.02, ...]
		self.decoder = nn.Linear(hidden_size, output_size)
	
	def forward(self, input, hidden):
		"""
		Fait passer les données à travers le réseau (forward pass)
		
		Args:
			input: les caractères d'entrée (convertis en indices)
			hidden: l'état mémoire du LSTM (h, c)
		
		Returns:
			output: probabilités pour chaque caractère possible
			hidden: nouvel état mémoire mis à jour
		"""
		seq_len = input.size(0)
		
		# 1. Encoder : caractère -> vecteur
		embedded = self.encoder(input.view(seq_len, 1))
		
		# 2. LSTM : traite la séquence avec sa mémoire
		output, hidden = self.lstm(embedded, hidden)
		
		# 3. Decoder : vecteur -> probabilités de caractères
		output = self.decoder(output.view(seq_len, -1))
		
		return output, hidden

	def init_hidden(self):
		"""
		Initialise la mémoire du LSTM à zéro
		
		Le LSTM a 2 types de mémoire :
		- h_0 : mémoire court terme (ce qu'on vient de voir)
		- c_0 : mémoire long terme (informations importantes retenues)
		"""
		h_0 = Variable(torch.zeros(self.n_layers, 1, self.hidden_size, device=device))
		c_0 = Variable(torch.zeros(self.n_layers, 1, self.hidden_size, device=device))
		return (h_0, c_0)

# =============================================================================
# FONCTIONS DE TRAITEMENT DES DONNÉES
# =============================================================================

def char_tensor(string):
	"""
	Convertit une chaîne de caractères en tenseur d'indices
	
	Exemple: "ABC" -> tensor([36, 37, 38])
	
	Pourquoi ? Le réseau ne comprend que les nombres, pas les lettres
	"""
	tensor = torch.zeros(len(string)).long()
	for c in range(len(string)):
		try:
			tensor[c] = all_characters.index(string[c])
		except:
			continue
	return Variable(tensor)

def random_training_set(corpus):
	"""
	Prend un morceau aléatoire du texte pour l'entraînement
	
	Retourne:
		inp: les caractères d'entrée (ex: "Hello Worl")
		target: les caractères attendus (ex: "ello World")
		
	Le modèle apprend que 'H' doit prédire 'e', 'e' doit prédire 'l', etc.
	"""
	corpus_len = len(corpus)
	start_index = random.randint(0, corpus_len - chunk_len - 1)
	end_index = start_index + chunk_len + 1
	chunk = corpus[start_index:end_index]
	
	inp = char_tensor(chunk[:-1]).to(device)     # Tous sauf le dernier
	target = char_tensor(chunk[1:]).to(device)   # Tous sauf le premier
	return inp, target

def compute_loss(decoder, corpus, criterion, num_samples=50):
	"""
	Calcule l'erreur moyenne du modèle sur un corpus (validation ou test)
	
	On prend plusieurs échantillons aléatoires pour avoir une bonne estimation
	"""
	decoder.eval()  # Mode évaluation (désactive certaines optimisations)
	total_loss = 0
	
	with torch.no_grad():  # Pas besoin de calculer les gradients ici
		for _ in range(num_samples):
			inp, target = random_training_set(corpus)
			hidden = decoder.init_hidden()
			output, _ = decoder(inp, hidden)
			loss = criterion(output, target)
			total_loss += loss.item()
	
	decoder.train()  # Retour en mode entraînement
	return total_loss / num_samples

def compute_perplexity(loss):
	"""
	Calcule la perplexité à partir de la loss
	
	Perplexité = exp(loss)
	
	Interprétation :
	- Perplexité = 10 : le modèle hésite entre ~10 caractères possibles
	- Perplexité = 2 : le modèle est presque sûr (2 choix possibles)
	
	Plus c'est bas, mieux c'est !
	"""
	return math.exp(loss)

# =============================================================================
# FONCTION D'ENTRAÎNEMENT (une itération)
# =============================================================================

def train(inp, target, decoder, decoder_optimizer, criterion):
	"""
	Effectue UNE étape d'apprentissage
	
	Étapes :
	1. Initialiser la mémoire
	2. Le modèle fait sa prédiction
	3. On calcule l'erreur (loss)
	4. On calcule comment corriger les poids (backpropagation)
	5. On applique la correction
	"""
	# 1. Initialiser la mémoire du LSTM
	hidden = decoder.init_hidden()
	
	# Remettre les gradients à zéro (sinon ils s'accumulent)
	decoder.zero_grad()
	
	# 2. Forward pass : le modèle prédit
	output, hidden = decoder(inp, hidden)
	
	# 3. Calcul de l'erreur entre prédiction et réalité
	loss = criterion(output, target)

	# 4. Backpropagation : on calcule comment corriger
	loss.backward()
	
	# 5. On applique les corrections aux poids
	decoder_optimizer.step()
	
	return loss.item()

# =============================================================================
# FONCTION DE GÉNÉRATION DE TEXTE
# =============================================================================

def evaluate(decoder, prime_str='A', predict_len=100, temperature=0.8):
	"""
	Génère du texte à partir d'un début de phrase
	
	Args:
		prime_str: le début de phrase (ex: "To be or ")
		predict_len: combien de caractères générer
		temperature: contrôle la créativité
			- 0.5 = conservateur, prévisible
			- 0.8 = équilibré (par défaut)
			- 1.2 = créatif, parfois incohérent
	"""
	hidden = decoder.init_hidden()
	prime_input = char_tensor(prime_str).to(device)
	predicted = prime_str

	# Phase de "priming" : on construit la mémoire avec le début de phrase
	for p in range(len(prime_str) - 1):
		_, hidden = decoder(prime_input[p].unsqueeze(0), hidden)
	
	inp = prime_input[-1]
	
	# Génération caractère par caractère
	for p in range(predict_len):
		# Le modèle prédit les probabilités du prochain caractère
		output, hidden = decoder(inp.unsqueeze(0), hidden)
		
		# Application de la température et échantillonnage
		output_dist = output.data.view(-1).div(temperature).exp()
		top_i = torch.multinomial(output_dist, 1)[0]
		
		# On ajoute le caractère prédit
		predicted_char = all_characters[top_i]
		predicted += predicted_char
		
		# Le caractère prédit devient l'entrée suivante
		inp = char_tensor(predicted_char).to(device)

	return predicted

# =============================================================================
# BOUCLE D'ENTRAÎNEMENT COMPLÈTE
# =============================================================================

def training(n_epochs, train_corpus, val_corpus, decoder, decoder_optimizer, criterion, args):
	"""
	Entraîne le modèle sur plusieurs milliers d'itérations (epochs)
	
	À chaque epoch :
	1. On prend un morceau aléatoire du texte d'entraînement
	2. On fait une étape d'apprentissage
	3. Toutes les 500 epochs, on vérifie sur le validation set
	4. Si c'est le meilleur score, on sauvegarde le modèle
	"""
	print()
	print('-----------')
	print('|  TRAIN  |')
	print('-----------')
	print()
	
	start = time.time()
	loss_avg = 0
	best_val_loss = float('inf')  # On commence avec une loss infinie
	print_every = 100             # Afficher toutes les 100 epochs
	validate_every = 500          # Vérifier sur validation toutes les 500 epochs
	
	for epoch in range(1, n_epochs + 1):
		# 1. Prendre un morceau aléatoire et s'entraîner dessus
		inp, target = random_training_set(train_corpus)
		loss = train(inp, target, decoder, decoder_optimizer, criterion)
		loss_avg += loss
		
		# Affichage périodique
		if epoch % print_every == 0:
			avg_loss = loss_avg / print_every
			print('[%s (%d %d%%) Train Loss: %.4f]' % (time_since(start), epoch, epoch / n_epochs * 100, avg_loss))
			loss_avg = 0
		
		# Évaluation sur le validation set
		if epoch % validate_every == 0:
			val_loss = compute_loss(decoder, val_corpus, criterion)
			val_ppl = compute_perplexity(val_loss)
			print(f'  [Validation] Loss: {val_loss:.4f} | Perplexité: {val_ppl:.2f}')
			
			# Sauvegarde du meilleur modèle
			if val_loss < best_val_loss:
				best_val_loss = val_loss
				modelFile = args.run + "_" + str(args.num_layers) + "_" + str(args.hidden_size) + ".pt"
				torch.save(decoder, join(args.model, modelFile))
				print(f'  -> Meilleur modèle sauvegardé! (Val Loss: {best_val_loss:.4f})')
	
	return best_val_loss

# =============================================================================
# MODE INTERACTIF DE GÉNÉRATION
# =============================================================================

def evaluating(decoder, length):
	"""
	Boucle interactive : l'utilisateur entre un début, le modèle génère la suite
	Appuyer sur Ctrl+C pour quitter
	"""
	print()
	print('------------')
	print('|   EVAL   |')
	print('------------')
	print()
	
	try:
		while True:
			print('Enter a starting string:')
			input1 = input()
			print()
			if len(input1) > 0:
				print('Generated', length, 'characters:')
				print(evaluate(decoder=decoder, prime_str=input1, predict_len=length, temperature=0.8))
			else:
				print('Input too short')
			print('------------')
			print()
			
	except KeyboardInterrupt:
		print("\nEvaluation terminated.")
		print('------------')

def read_passwords(path_file):
	with open(path_file, "r", encoding="utf-8", errors="ignore") as f:
		return [line.rstrip("\n\r") for line in f if line.rstrip("\n\r")]

def clean_generated_password(text, min_len, max_len):
	pwd = text.split("\n", 1)[0].replace("\r", "").replace("\t", "").strip()
	if len(pwd) < min_len:
		return None
	return pwd[:max_len]

def generate_password(decoder, min_len, max_len, temperature):
	seed_chars = string.ascii_letters + string.digits
	for _ in range(10):
		prime = random.choice(seed_chars)
		raw = evaluate(decoder=decoder, prime_str=prime, predict_len=max_len + 4, temperature=temperature)
		pwd = clean_generated_password(raw, min_len, max_len)
		if pwd:
			return pwd
	return ''.join(random.choice(seed_chars) for _ in range(min_len))

def testing_passwords(decoder, eval_data, samples, min_len, max_len, temperature):
	eval_passwords = read_passwords(eval_data)
	eval_set = set(eval_passwords)
	unique_generated = set()
	total_hits = 0
	unique_hits = set()

	for _ in range(samples):
		pwd = generate_password(decoder, min_len, max_len, temperature)
		unique_generated.add(pwd)
		if pwd in eval_set:
			total_hits += 1
			unique_hits.add(pwd)

	hit_rate = (total_hits / samples) * 100 if samples else 0
	coverage = (len(unique_hits) / len(eval_set)) * 100 if eval_set else 0
	unique_ratio = (len(unique_generated) / samples) * 100 if samples else 0

	print()
	print('--------------------------')
	print('| PASSWORD TEST SUMMARY |')
	print('--------------------------')
	print('Eval set size:', len(eval_set))
	print('Samples generated:', samples)
	print('Unique generated:', len(unique_generated), '(%.2f%%)' % unique_ratio)
	print('Hit rate (generated in eval): %.4f%%' % hit_rate)
	print('Coverage (eval passwords found): %.4f%%' % coverage)
	print('Unique hits:', len(unique_hits))

	return {
		"eval_size": len(eval_set),
		"samples": samples,
		"unique_generated": len(unique_generated),
		"hit_rate_percent": hit_rate,
		"coverage_percent": coverage,
		"unique_hits": len(unique_hits)
	}

# =============================================================================
# POINT D'ENTRÉE PRINCIPAL
# =============================================================================

if __name__ == '__main__':
	
	# Configuration des arguments en ligne de commande
	parser = ArgumentParser()
	parser.add_argument("-d", "--trainingData", default="data/shakespeare.txt", type=str,
	                    help="Fichier texte pour l'entraînement")
	parser.add_argument("-te", "--trainEval", default='train', type=str,
	                    help="Mode: 'train', 'eval' ou 'test'")
	parser.add_argument("-r", "--run", default="lstmOptimized", type=str,
	                    help="Nom du modèle")
	parser.add_argument("-m", "--model", default='models', type=str,
	                    help="Dossier de sauvegarde")
	parser.add_argument('--length', default=100, type=int,
	                    help="Longueur du texte généré")
	parser.add_argument('--num_layers', default=3, type=int,
	                    help="Nombre de couches LSTM (profondeur)")
	parser.add_argument('--hidden_size', default=256, type=int,
	                    help="Taille de la mémoire (capacité)")
	parser.add_argument('--max_epochs', default=20000, type=int,
	                    help="Nombre d'itérations d'entraînement")
	parser.add_argument('--evalData', default='TrainEval/eval.txt', type=str,
	                    help="path to evaluation password file")
	parser.add_argument('--samples', default=5000, type=int,
	                    help="number of generated passwords during test mode")
	parser.add_argument('--pwd_min_len', default=6, type=int,
	                    help="minimum generated password length")
	parser.add_argument('--pwd_max_len', default=16, type=int,
	                    help="maximum generated password length")
	parser.add_argument('--temperature', default=0.8, type=float,
	                    help="sampling temperature")
	
	args = parser.parse_args()

	# Lecture du fichier texte et conversion en ASCII
	file = unidecode.unidecode(open(args.trainingData).read())
	file_len = len(file)

	# Division du corpus en train/validation/test
	train_corpus, val_corpus, test_corpus = split_corpus(file)

	# Création du modèle LSTM
	decoder = RNN(n_characters, args.hidden_size, n_characters, args.num_layers).to(device)
	
	# Optimiseur Adam : ajuste automatiquement le taux d'apprentissage
	decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.002)
	
	# Fonction de perte : mesure l'erreur de prédiction
	criterion = nn.CrossEntropyLoss()

	print(f'\nTotal corpus: {file_len:,} chars')
	print(f'Model: LSTM (Layers: {args.num_layers}, Hidden: {args.hidden_size})')

	modelFile = args.run + "_" + str(args.num_layers) + "_" + str(args.hidden_size) + ".pt"

	# Créer le dossier de modèles si nécessaire
	if not path.exists(args.model):
		makedirs(args.model)
	
	# ==========================================================================
	# MODE ENTRAÎNEMENT
	# ==========================================================================
	if args.trainEval == 'train':
		decoder.train()  # Active le mode entraînement
		best_val_loss = training(args.max_epochs, train_corpus, val_corpus, decoder, decoder_optimizer, criterion, args)
		
		# Évaluation finale sur le test set (données jamais vues)
		print()
		print('------------')
		print('|   TEST   |')
		print('------------')
		
		# Charger le meilleur modèle sauvegardé
		decoder = torch.load(join(args.model, modelFile), weights_only=False)
		decoder.eval()  # Mode évaluation
		
		# Calculer les métriques finales
		test_loss = compute_loss(decoder, test_corpus, criterion, num_samples=100)
		test_ppl = compute_perplexity(test_loss)
		
		print(f'Test Loss: {test_loss:.4f}')
		print(f'Test Perplexité: {test_ppl:.2f}')
		print()
		print(f"Training Complete. Best model saved to {modelFile}")
	
	# ==========================================================================
	# MODE GÉNÉRATION
	# ==========================================================================
	elif args.trainEval == 'eval':
		# Charger le modèle entraîné
		decoder = torch.load(join(args.model, modelFile), weights_only=False)
		decoder.eval()
		decoder.to(device)
		# Lancer le mode interactif
		evaluating(decoder, args.length)
	elif args.trainEval == 'test':
		decoder = torch.load(join(args.model, modelFile), weights_only=False)
		decoder.eval().to(device)
		testing_passwords(
			decoder=decoder,
			eval_data=args.evalData,
			samples=args.samples,
			min_len=args.pwd_min_len,
			max_len=args.pwd_max_len,
			temperature=args.temperature
		)
	else:
		print('Choose trainEval option (--trainEval train/eval/test)')
