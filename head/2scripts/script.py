import random
import pandas as pd
import json
from collections import Counter
import seaborn as sns
from collections import defaultdict
import numpy as np


random.seed(42)


def softmax(x):
       return np.exp(x - x.max()) / np.sum(np.exp(x - np.max(x)))

def fromAA_toDNA(proteins):
       codon_dict = {
              'DNA': ['TTT', 'TTC', 'TTA', 'TTG', 'TCT', 'TCC', 'TCA', 'TCG', 'TAT', 'TAC', 'TAA', 'TAG', 'TGT', 'TGC',
                      'TGA', 'TGG', 'CTT', 'CTC', 'CTA', 'CTG', 'CCT', 'CCC', 'CCA', 'CCG', 'CAT', 'CAC', 'CAA', 'CAG',
                      'CGT', 'CGC', 'CGA', 'CGG', 'ATT', 'ATC', 'ATA', 'ATG', 'ACT', 'ACC', 'ACA', 'ACG', 'AAT', 'AAC',
                      'AAA', 'AAG', 'AGT', 'AGC', 'AGA', 'AGG', 'GTT', 'GTC', 'GTA', 'GTG', 'GCT', 'GCC', 'GCA', 'GCG',
                      'GAT', 'GAC', 'GAA', 'GAG', 'GGT', 'GGC', 'GGA', 'GGG'],
              'AA': ['F', 'F', 'L', 'L', 'S', 'S', 'S', 'S', 'Y', 'Y', '_', '_', 'C', 'C', '_', 'W', 'L', 'L', 'L', 'L',
                     'P', 'P', 'P', 'P', 'H', 'H', 'Q', 'Q', 'R', 'R', 'R', 'R', 'I', 'I', 'I', 'M', 'T', 'T', 'T', 'T',
                     'N', 'N', 'K', 'K', 'S', 'S', 'R', 'R', 'V', 'V', 'V', 'V', 'A', 'A', 'A', 'A', 'D', 'D', 'E', 'E',
                     'G', 'G', 'G', 'G']}
       r2a = dict(zip(codon_dict['DNA'], codon_dict['AA'])) # Создал библиотеку соотношений кодонов и аминокислот
       some_dna = []
       codon = []
       for i in proteins: # В цикле происходит превращение, аминоксилоты из переменной covid_proteins заменяются кодонами
              for j in i:
                     for k, v in r2a.items():
                            if j == v:
                                   codon.append(k)
                     if codon != []:
                            some_dna.append(random.choice(codon)) # Если подходящих кодонов несколько, то рандомным из них
                     codon = []
       return(some_dna)

def nuc_mut_counter(mutations): # функция принимает на вход табличку с мутациями и подсчитывает их вероятности
       # на выходе выдаёт табличку из чего во что мутирует и с какой вероятностью
       nuclear_mutations = []
       for i in mutations:
              mutation = json.loads(i)
              if 'nuc' in mutation:
                     nuc_mutations = mutation['nuc']
                     for mut in nuc_mutations:
                            nuclear_mutations.append((mut[0], mut[-1]))

       first = []
       second = []
       probability = []
       summ_A = 0
       summ_T = 0
       summ_G = 0
       summ_C = 0

       nucs, nuc_counts = np.unique(list(map(lambda x: x[0], nuclear_mutations)), return_counts=True)
       nuc_stat = dict(zip(nucs, nuc_counts))
       # for k,v in Counter(nuclear_mutations).items(): # Нормализую данные по колличеству мутаций по нуклеотидам
       #        if k[0] == 'A':
       #               summ_A += v
       #        if k[0] == 'T':
       #               summ_T += v
       #        if k[0] == 'G':
       #               summ_G += v
       #        if k[0] == 'C':
       #               summ_C += v
       # for k,v in Counter(nuclear_mutations).items():
       #        first.append(k[0])
       #        second.append(k[1])
       #        if k[0] == 'A':
       #               probability.append(v/summ_A)
       #        if k[0] == 'T':
       #               probability.append(v/summ_T)
       #        if k[0] == 'G':
       #               probability.append(v/summ_G)
       #        if k[0] == 'C':
       #               probability.append(v/summ_C)
       mutations, mutation_counts = np.unique(nuclear_mutations, return_counts=True, axis=0)


       mutation_probability = pd.DataFrame(
              {'From': list(map(lambda x:x[0], mutations)),
               'To': list(map(lambda x:x[-1], mutations)),
               'probability': mutation_counts})

       mutation_probability['probability'] = \
              mutation_probability.apply(
                     lambda x: x['probability']/ nuc_stat[x['From']],
                                         axis=1)

       return(mutation_probability)

def mutation(dna, mutation_probability): # Функция вносит мутации в рандомный кодон, рандомный нуклеотид из ДНК с определённой вероятностью
       # На выходе имеем переменную с числами означающими сколько поколений прожил вирус без возникновения мутаций, меняющих аминокислоту
       codon_dict = {
              'DNA': ['TTT', 'TTC', 'TTA', 'TTG', 'TCT', 'TCC', 'TCA', 'TCG', 'TAT', 'TAC', 'TAA', 'TAG', 'TGT', 'TGC',
                      'TGA', 'TGG', 'CTT', 'CTC', 'CTA', 'CTG', 'CCT', 'CCC', 'CCA', 'CCG', 'CAT', 'CAC', 'CAA', 'CAG',
                      'CGT', 'CGC', 'CGA', 'CGG', 'ATT', 'ATC', 'ATA', 'ATG', 'ACT', 'ACC', 'ACA', 'ACG', 'AAT', 'AAC',
                      'AAA', 'AAG', 'AGT', 'AGC', 'AGA', 'AGG', 'GTT', 'GTC', 'GTA', 'GTG', 'GCT', 'GCC', 'GCA', 'GCG',
                      'GAT', 'GAC', 'GAA', 'GAG', 'GGT', 'GGC', 'GGA', 'GGG'],
              'AA': ['F', 'F', 'L', 'L', 'S', 'S', 'S', 'S', 'Y', 'Y', '_', '_', 'C', 'C', '_', 'W', 'L', 'L', 'L', 'L',
                     'P', 'P', 'P', 'P', 'H', 'H', 'Q', 'Q', 'R', 'R', 'R', 'R', 'I', 'I', 'I', 'M', 'T', 'T', 'T', 'T',
                     'N', 'N', 'K', 'K', 'S', 'S', 'R', 'R', 'V', 'V', 'V', 'V', 'A', 'A', 'A', 'A', 'D', 'D', 'E', 'E',
                     'G', 'G', 'G', 'G']}
       r2a = dict(zip(codon_dict['DNA'], codon_dict['AA']))  # Создал библиотеку соотношений кодонов и аминокислот
       new_DNA = dna
       mutation_counter = 0
       mutation_list = []
       A_prability = []
       G_prability = []
       T_prability = []
       C_prability = []
       A_to = []
       T_to = []
       G_to = []
       C_to = []
       for k in range(12):
              if mutation_probability.From[k] == 'A':
                     A_prability.append(mutation_probability.probability[k])
                     A_to.append(mutation_probability.To[k])
              if mutation_probability.From[k] == 'T':
                     T_prability.append(mutation_probability.probability[k])
                     T_to.append(mutation_probability.To[k])
              if mutation_probability.From[k] == 'G':
                     G_prability.append(mutation_probability.probability[k])
                     G_to.append(mutation_probability.To[k])
              if mutation_probability.From[k] == 'C':
                     C_prability.append(mutation_probability.probability[k])
                     C_to.append(mutation_probability.To[k])
       for i in range(100000):
              random_codon = random.choice(new_DNA) # Беру случайный кодон из ДНК
              AA = r2a[random_codon] # Присваиваю ему аминокислоту
              index_random_nucl = random.choice(range(len(random_codon)))
              random_nucl = random_codon[index_random_nucl] # Беру по индексу рандомный нуклеотид
              for k in range(12):
                     if random_nucl == mutation_probability.From[k]:  # Проверяю совпадает ли нуклеотид с таковыми из таблички с мутациями
                            if mutation_probability.From[k] == 'A':
                                   new_nucl = np.random.choice(A_to, p = A_prability)
                                   new_codon = random_codon[:index_random_nucl] + new_nucl + random_codon[index_random_nucl + 1:]
                            if mutation_probability.From[k] == 'T':
                                   new_nucl = np.random.choice(T_to, p = T_prability)
                                   new_codon = random_codon[:index_random_nucl] + new_nucl + random_codon[index_random_nucl + 1:]
                            if mutation_probability.From[k] == 'G':
                                   new_nucl = np.random.choice(G_to, p = G_prability)
                                   new_codon = random_codon[:index_random_nucl] + new_nucl + random_codon[index_random_nucl + 1:]
                            if mutation_probability.From[k] == 'C':
                                   new_nucl = np.random.choice(C_to, p = C_prability)
                                   new_codon = random_codon[:index_random_nucl] + new_nucl + random_codon[index_random_nucl + 1:]
              AA_new = r2a[new_codon] # Присвоил новому кодону аминокислоту
              if AA == AA_new: # Сравниваю аминокислоты
                     mutation_counter += 1  # Если они совпадают - добавляю к счётчику мутаций еденичку
              else:
                     mutation_list.append(mutation_counter)
                     mutation_counter = 0
       return(mutation_list)

def distribution_of_mutations(mutation_list):
       data = pd.DataFrame(mutation_list, columns = ['number of mutation without changing amino acid'])
       # Рисую график распределения
       sns.set()
       sns_plot = sns.distplot(data['number of mutation without changing amino acid'])
       fig = sns_plot.get_figure()
       fig.savefig('data/covid_mutation_distribution.png') # Сохранил график в файл

def codon_usage(dna, mutation_probability):
       codon_usage_dict = {'TTT' : [0.5], 'TTC' : [0.5], 'TTA' : [1/6], 'TTG' : [1/6], 'TCT' : [0.25], 'TCC' : [0.25], 'TCA' : [0.25], 'TCG' : [0.25], 'TAT' : [0.5], 'TAC' : [0.5], 'TGT' : [0.5], 'TGC' : [0.5],
                      'TGG' : [1], 'CTT' : [1/6], 'CTC' : [1/6], 'CTA' : [1/6], 'CTG' : [1/6], 'CCT' : [0.25], 'CCC' : [0.25], 'CCA' : [0.25], 'CCG' : [0.25], 'CAT' : [0.5], 'CAC' : [0.5], 'CAA' : [0.5], 'CAG' : [0.5],
                      'CGT' : [0.25], 'CGC' : [0.25], 'CGA' : [0.25], 'CGG' : [0.25], 'ATT' : [1/3], 'ATC' : [1/3], 'ATA' : [1/3], 'ATG' : [1], 'ACT' : [0.25], 'ACC' : [0.25], 'ACA' : [0.25], 'ACG' : [0.25], 'AAT' : [0.5], 'AAC' : [0.5],
                      'AAA' : [0.5], 'AAG' : [0.5], 'AGT' : [0.5], 'AGC' : [0.5], 'AGA' : [0.5], 'AGG' : [0.5], 'GTT' : [0.25], 'GTC' : [0.25], 'GTA' : [0.25], 'GTG' : [0.25], 'GCT' : [0.25], 'GCC' : [0.25], 'GCA' : [0.25], 'GCG' : [0.25],
                      'GAT' : [0.5], 'GAC' : [0.5], 'GAA' : [0.5], 'GAG' : [0.5], 'GGT' : [0.25], 'GGC' : [0.25], 'GGA' : [0.25], 'GGG' : [0.25]}
       for k in codon_usage_dict:
              codon_usage_dict[k] = []

       newDNA = dna
       codon_dict = {
              'DNA': ['TTT', 'TTC', 'TTA', 'TTG', 'TCT', 'TCC', 'TCA', 'TCG', 'TAT', 'TAC', 'TAA', 'TAG', 'TGT', 'TGC',
                      'TGA', 'TGG', 'CTT', 'CTC', 'CTA', 'CTG', 'CCT', 'CCC', 'CCA', 'CCG', 'CAT', 'CAC', 'CAA', 'CAG',
                      'CGT', 'CGC', 'CGA', 'CGG', 'ATT', 'ATC', 'ATA', 'ATG', 'ACT', 'ACC', 'ACA', 'ACG', 'AAT', 'AAC',
                      'AAA', 'AAG', 'AGT', 'AGC', 'AGA', 'AGG', 'GTT', 'GTC', 'GTA', 'GTG', 'GCT', 'GCC', 'GCA', 'GCG',
                      'GAT', 'GAC', 'GAA', 'GAG', 'GGT', 'GGC', 'GGA', 'GGG'],
              'AA': ['F', 'F', 'L', 'L', 'S', 'S', 'S', 'S', 'Y', 'Y', '_', '_', 'C', 'C', '_', 'W', 'L', 'L', 'L', 'L',
                     'P', 'P', 'P', 'P', 'H', 'H', 'Q', 'Q', 'R', 'R', 'R', 'R', 'I', 'I', 'I', 'M', 'T', 'T', 'T', 'T',
                     'N', 'N', 'K', 'K', 'S', 'S', 'R', 'R', 'V', 'V', 'V', 'V', 'A', 'A', 'A', 'A', 'D', 'D', 'E', 'E',
                     'G', 'G', 'G', 'G']}
       r2a = dict(zip(codon_dict['DNA'], codon_dict['AA']))
       a2codon = defaultdict(list)
       A_prability = []
       G_prability = []
       T_prability = []
       C_prability = []
       A_to = []
       T_to = []
       G_to = []
       C_to = []
       for k in range(12):
              if mutation_probability.From[k] == 'A':
                     A_prability.append(mutation_probability.probability[k])
                     A_to.append(mutation_probability.To[k])
              if mutation_probability.From[k] == 'T':
                     T_prability.append(mutation_probability.probability[k])
                     T_to.append(mutation_probability.To[k])
              if mutation_probability.From[k] == 'G':
                     G_prability.append(mutation_probability.probability[k])
                     G_to.append(mutation_probability.To[k])
              if mutation_probability.From[k] == 'C':
                     C_prability.append(mutation_probability.probability[k])
                     C_to.append(mutation_probability.To[k])
       for key, val in sorted(r2a.items()):
              a2codon[val].append(key)
       for i in range(1000):
              index_random_codon = random.choice(range(len(newDNA)))
              random_codon = newDNA[index_random_codon]# Беру случайный кодон из ДНК
              AA = r2a[random_codon]  # Присваиваю ему аминокислоту
              index_random_nucl = random.choice(range(len(random_codon)))
              random_nucl = random_codon[index_random_nucl]# Беру по индексу рандомный нуклеотид



              new_nucl = np.random.choice(mutation_probability[mutation_probability.From == random_nucl].To,
                                          p=mutation_probability[mutation_probability.From == random_nucl].probability)
              new_codon = random_codon[:index_random_nucl] + new_nucl + random_codon[index_random_nucl + 1:]

              # for k in range(12):
              #        if random_nucl == mutation_probability.From[k]:  # Проверяю совпадает ли нуклеотид с таковыми из таблички с мутациями
              #               if mutation_probability.From[k] == 'A':
              #                      new_nucl = np.random.choice(A_to, p = A_prability)
              #                      new_codon = random_codon[:index_random_nucl] + new_nucl + random_codon[index_random_nucl + 1:]
              #               if mutation_probability.From[k] == 'T':
              #                      new_nucl = np.random.choice(T_to, p = T_prability)
              #                      new_codon = random_codon[:index_random_nucl] + new_nucl + random_codon[index_random_nucl + 1:]
              #               if mutation_probability.From[k] == 'G':
              #                      new_nucl = np.random.choice(G_to, p = G_prability)
              #                      new_codon = random_codon[:index_random_nucl] + new_nucl + random_codon[index_random_nucl + 1:]
              #               if mutation_probability.From[k] == 'C':
              #                      new_nucl = np.random.choice(C_to, p = C_prability)
              #                      new_codon = random_codon[:index_random_nucl] + new_nucl + random_codon[index_random_nucl + 1:]
              AA_new = r2a[new_codon]  # Присвоил новому кодону аминокислоту
              if AA_new == AA:
                     newDNA[index_random_codon] = new_codon
              if i % 100 == 0:
                     for codon, count in Counter(newDNA).items():
                            for codon_data, count_data in codon_usage_dict.items():
                                   if codon == codon_data:
                                          count_data.append(count)
       codon_usage_data = pd.DataFrame.from_dict(codon_usage_dict)
       res = []
       for idx, row in codon_usage_data.iterrows():
              for aa, codons in a2codon.items():
                     if aa != '_':
                            row[codons] = row[codons] / row[codons].sum()
              res.append(row)

       norm_codon_usage = pd.DataFrame(res) #codon_usage_data.apply(softmax, axis=1)

       # codon_usage_data
       # for z in a2codon.values(): # Не работает(((
       #        z_summ = []
       #        for i in range(len(z)):
       #               print(i)
       #               z_summ.append(z[i])
       #        for i in range(len(z)):
       #               print(z_summ)
       #               norm_codon_usage[[z][i]] = codon_usage_data[[z][i]] / codon_usage_data[z_summ].sum(axis=0)
       return(norm_codon_usage)


if __name__ == '__main__':
       data = open('data/covid_proteins.txt', 'r')
       covid_proteins = data.read()
       # открываю файл со скопированной последовательностью аминокислот слепленных в один файл
       covid_dna = fromAA_toDNA(covid_proteins)
       #print(covid_dna) # Тут хранится последовательность ДНК

       # Далее считаю вероятность мутаций

       covid_data = pd.read_csv('data/covid_next_strain.tsv', sep = '\t') # Открываю файл табличку в которой есть вероятности
       only_mutations = covid_data['mutations'].dropna() # Слегка обрабатываю
       only_mutations = only_mutations.str.replace("'", '"')
       mutation_probability = nuc_mut_counter(only_mutations)
       mutation_probability.to_csv('data/mutation_probability.tsv', sep = '\t')
       # Сохранил вероятности мутаций в файл

       # Далее на основании полученных данных мутируем нашу ДНК и подсчитываем количество мутаций
       #mutation_list = mutation(covid_dna, mutation_probability)
       #print(mutation_list) # Тут хранятся данные по количеству мутаций без изменения аминокислоты

       #distribution_of_mutations(mutation_list) # Строим график распределения

       my_codon_usage = codon_usage(covid_dna, mutation_probability)
       my_codon_usage.to_csv('data/codon_usage_data.tsv', sep = '\t')
       print(my_codon_usage.head())
       print(my_codon_usage.tail())
