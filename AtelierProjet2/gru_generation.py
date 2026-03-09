import unidecode
import string
import random
import re

from os import listdir, path, makedirs, popen
from os.path import isdir, isfile, join

import torch
import torch.nn as nn
from torch.autograd import Variable

import time, math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from argparse import ArgumentParser

if torch.cuda.is_available():
	device = torch.device("cuda:0")
	print('CUDA AVAILABLE')
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
	device = torch.device("mps")
	print('MPS (APPLE SILICON GPU) AVAILABLE')
else:
	device = torch.device("cpu")
	print('ONLY CPU AVAILABLE')
	
all_characters = string.printable
n_characters = len(all_characters)
chunk_len = 13

n_epochs = 200000
print_every = 10
plot_every = 10
hidden_size = 512
n_layers = 3
lr = 0.005

def random_chunk(file):
	start_index = random.randint(0, file_len - chunk_len)
	end_index = start_index + chunk_len + 1
	return file[start_index:end_index]
		
# Turn string into list of longs
def char_tensor(string):
	tensor = torch.zeros(len(string)).long()
	for c in range(len(string)):
		tensor[c] = all_characters.index(string[c])
	return Variable(tensor)

def random_training_set(file):    
	chunk = random_chunk(file)
	inp = char_tensor(chunk[:-1]).to(device)
	target = char_tensor(chunk[1:]).to(device)
	return inp, target
	
def evaluate(decoder, prime_str='A', predict_len=100, temperature=0.8):
	hidden = decoder.init_hidden()
	prime_input = char_tensor(prime_str).to(device)
	predicted = prime_str

	# Use priming string to "build up" hidden state
	for p in range(len(prime_str) - 1):
		_, hidden = decoder(prime_input[p], hidden)
	inp = prime_input[-1]
	
	for p in range(predict_len):
		output, hidden = decoder(inp, hidden)
		
		# Sample from the network as a multinomial distribution
		output_dist = output.data.view(-1).div(temperature).exp()
		top_i = torch.multinomial(output_dist, 1)[0]
		
		# Add predicted character to string and use as next input
		predicted_char = all_characters[top_i]
		predicted += predicted_char
		inp = char_tensor(predicted_char).to(device)

	return predicted
	
def time_since(since):
	s = time.time() - since
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)

def train(inp, target):
	hidden = decoder.init_hidden()
	decoder.zero_grad()
	loss = 0
	for c in range(inp.size(0)): #range(chunk_len):
		output, hidden = decoder(inp[c], hidden)
		loss += criterion(output, target[c].unsqueeze(0))

	loss.backward()
	decoder_optimizer.step()
	
	return loss.item() / chunk_len
	
class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, n_layers=1):
		super(RNN, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers
		
		self.encoder = nn.Embedding(input_size, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
		self.decoder = nn.Linear(hidden_size, output_size)
	
	def forward(self, input, hidden):
		input = self.encoder(input.view(1, -1))
		output, hidden = self.gru(input.view(1, 1, -1), hidden)
		output = self.decoder(output.view(1, -1))
		return output, hidden

	def init_hidden(self):
		return Variable(torch.zeros(self.n_layers, 1, self.hidden_size, device=device))

def training(n_epochs, file):
	print()
	print('-----------')
	print('|  TRAIN  |')
	print('-----------')
	print()
	
	start = time.time()
	all_losses = []
	loss_avg = 0
	best_loss = 100
	print_every = n_epochs / 100
	
	for epoch in range(1, n_epochs + 1):
		loss = train(*random_training_set(file))       
		loss_avg += loss
		
		if epoch % print_every == 0:
			print('[%s (%d %d%%) %.4f (%.4f)]' % (time_since(start), epoch, epoch / n_epochs * 100, loss_avg / epoch, loss))
			
		if best_loss > (loss_avg / epoch):
			best_loss = loss_avg / epoch
			print('[%s (%d %d%%) %.4f (%.4f)]' % (time_since(start), epoch, epoch / n_epochs * 100, loss_avg / epoch, loss))
			#print(evaluate('Wh', 100), '\n')

		#if epoch % plot_every == 0:
		#	all_losses.append(loss_avg / plot_every)
		#	loss_avg = 0
			
	#plt.figure()
	#plt.plot(all_losses)

def evaluating(decoder, length):
	print()
	print('------------')
	print('|   EVAL   |')
	print('------------')
	print()
	
	try:
		while True:
			print('Enter a starting two or tree charachters')
			input1 = input()
			print()
			if len(input1) > 0:
				print('Generated ', length, 'charcaters: ')
				print(evaluate(decoder = decoder, prime_str = input1, predict_len = length, temperature = 0.8))
			else:
				print(input1, ' length < 1')
			print('------------')
			print()
			
	except KeyboardInterrupt:
		print("Press Ctrl-C to terminate evaluating")
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
	generated = []
	unique_generated = set()
	total_hits = 0
	unique_hits = set()

	for _ in range(samples):
		pwd = generate_password(decoder, min_len, max_len, temperature)
		generated.append(pwd)
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
	
if __name__ == '__main__':
	
	parser = ArgumentParser()
	#
	parser.add_argument("-d", "--trainingData", default="data/shakespeare.txt", type=str, help="trainingData [path/to/the/data]")
	parser.add_argument("-te", "--trainEval", default='train', type=str, help="trainEval [train, eval, test]")
	#
	parser.add_argument("-r", "--run", default="rnnGeneration", type=str, help="name of the model saved file")
	parser.add_argument("-m", "--model", default='models', type=str, help="model to save (train) or to load (eval) [path/to/the/model]")
	#
	parser.add_argument('--length', default=100, type=int, help="sequence length during eval process [< 1000]")
	parser.add_argument('--num_layers', default=2, type=int)
	parser.add_argument('--hidden_size', default=128, type=int)
	parser.add_argument('--max_epochs', default=10000, type=int)
	parser.add_argument('--evalData', default='TrainEval/eval.txt', type=str, help="path to evaluation password file")
	parser.add_argument('--samples', default=5000, type=int, help="number of generated passwords during test mode")
	parser.add_argument('--pwd_min_len', default=6, type=int, help="minimum generated password length")
	parser.add_argument('--pwd_max_len', default=16, type=int, help="maximum generated password length")
	parser.add_argument('--temperature', default=0.8, type=float, help="sampling temperature")
	#
	args = parser.parse_args()
	#
	repData = args.trainingData #"data/out/text10.txt"
	#repData = "data/shakespeare.txt"

	file = unidecode.unidecode(open(repData).read())
	file_len = len(file)

	decoder = RNN(n_characters, args.hidden_size, n_characters, args.num_layers).to(device)
	decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
	criterion = nn.CrossEntropyLoss()

	n_epochs = args.max_epochs

	print(random_chunk(file))

	print('Training file_len =', file_len)

	modelFile = args.run + "_" + str(args.num_layers) + "_" + str(args.hidden_size) + ".pt"

	if not path.exists(args.model):
		makedirs(args.model)
			
	if args.trainEval == 'train':
		decoder.train()
		training(n_epochs, file)
		torch.save(decoder, join(args.model, modelFile))
	elif args.trainEval == 'eval':
		decoder = torch.load(join(args.model, modelFile), weights_only=False)
		decoder.eval().to(device)
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
		print('Choose trainEval option (--trainEval train/eval/test')

	
