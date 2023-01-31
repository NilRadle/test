# Похоже, весь этот скрипт не нужен
import pandas as pd

# sum_codons = pd.read_csv('Codons.csv', sep='\t')

# codons = list(sum_codons.columns[1:]) # Список всех кодонов из таблицы
# for i in codons:                        # Перестановка кодонов в списке, чтобы A/G и C/T были соседними
#     if i.endswith('G'):                 # т.к. в изначальной таблице порядок кодонов не соответствует Fig. 4A
#         ind = codons.index(i)
#         replace = codons[ind]
#         codons[ind] = codons[ind-1]
#         codons[ind-1] = replace
#     else: continue                      # НЕ НУЖНО, все равно частоты идут на график(но может пригодиться для нормализации)

# frequency = pd.DataFrame(columns=codons) # Создаем таблицу со столбцами по кодонам и строкам по видам
# # frequency.loc[0] - частоты кодонов для наблюдаемого транскриптома
# frequency.loc[0] = ['0.0173841059602649', '0.0041390728476821195', '0.024006622516556293', '0.008554083885209713', '0.0358719646799117', '0.004690949227373068', '0.030077262693156734', '0.013245033112582781', '0.0', '0.0', '0.009381898454746136', '0.004415011037527594', '0.036699779249448124', '0.01379690949227373', '0.03145695364238411', '0.04415011037527594', '0.023178807947019868', '0.004415011037527594', '0.022902869757174392', '0.006622516556291391', '0.02455849889624724', '0.005518763796909493', '0.020695364238410598', '0.008002207505518763', '0.012417218543046357', '0.0027593818984547464', '0.0027593818984547464', '0.0016556291390728477', '0.060430463576158944', '0.016280353200883002', '0.026214128035320087', '0.02924944812362031', '0.020143487858719646', '0.004966887417218543', '0.014072847682119206', '0.00717439293598234', '0.03228476821192053', '0.004415011037527594', '0.03780353200883002', '0.012693156732891833', '0.025938189845474614', '0.012141280353200883', '0.015728476821192054', '0.00772626931567329', '0.024006622516556293', '0.008554083885209713', '0.01020971302428256', '0.012693156732891833', '0.0', '0.0', '0.01793598233995585', '0.010485651214128035', '0.021247240618101546', '0.0024834437086092716', '0.01545253863134658', '0.009105960264900662', '0.025386313465783666', '0.006622516556291391', '0.0033112582781456954', '0.0033112582781456954', '0.02759381898454746', '0.0030353200883002206', '0.03118101545253863', '0.02676600441501104']
# frequency_values = []    # Список частотных значений кодонов (будет обновляться для каждого вида)
# for i in [149]:  # В скобках число поколений-1 из Modulation.py                      # Нормализация
#     for j in range(1, len(sum_codons.columns)):
#         total = sum_codons.sum(axis=1)[i] # Складываем  кол-во кодонов
#         frequency_values.append(sum_codons.loc[i].values[j] / total)
#     frequency.loc[1] = frequency_values   # Добавляем в соответствующую стороку значения
#     frequency_values = []    # Обнуляем(хе) список
# frequency.to_csv('ModulatingCodonFrequencies.csv', sep='\t')



from scipy.stats import pearsonr            # Расчет корреляций для средних частот
from scipy.stats import spearmanr

number_of_generations = 150
species_list = ['Mus_musculus']  # ,'Abbottina_rivularis', 'Uroplatus_ebenaui', 'Eopsaltria_australis', 'Tylototriton_verrucosus'
samples_number = int(input('Количество образцов:'))
for p in range(len(species_list)):
    species = species_list[p]
    FrequenciesMean = pd.read_csv('Results\{}\{}\Frequencies.csv'.format(species, 0), sep='\t').copy().drop('Unnamed: 0', axis=1)
    codons = list(FrequenciesMean.columns.values)
    for q in range(1, samples_number):
        freq = pd.read_csv('Results\{}\{}\Frequencies.csv'.format(species, q), sep='\t').copy().drop('Unnamed: 0', axis=1)
        for i in codons:
            FrequenciesMean[i] += freq[i]
    for i in codons:
        FrequenciesMean[i] = FrequenciesMean[i] / samples_number
    FrequenciesMean.to_csv('Results\{}\FrequenciesMean.csv'.format(species), sep='\t')

    data = pd.read_csv('Results\{}\FrequenciesMean.csv'.format(species), sep ='\t').copy().drop('Unnamed: 0', axis=1)
    spearman = []
    pearson = []
    pv_s = []
    pv_p = []
    for i in data.index.values[1:]:
        data1, data2 = data.loc[0], data.loc[i]
        corrp, p_p = pearsonr(data1, data2)
        corrs, p_s = spearmanr(data1, data2)
        spearman.append(corrs)
        pearson.append(corrp)
        pv_s.append(p_s)
        pv_p.append(p_p)
    spear_corr = pd.DataFrame(index=[i for i in range(number_of_generations)]
                              , columns=['spearman', 'pv_s', 'pearson', 'pv_p'])
    spear_corr['spearman'] = spearman
    spear_corr['pearson'] = pearson
    spear_corr['pv_s'] = pv_s
    spear_corr['pv_p'] = pv_p
    spear_corr.to_csv('Results\{}\SpearmanMean.csv'.format(species), sep='\t')



# Abbottina_rivularis - рыба(Речная абботтина)
# Uroplatus_ebenaui  - ящерица(геккон)
# Eopsaltria_australis - птица(Зарянковая мухоловка
# Tylototriton_verrucosus - земноводное
# Mus_musculus - мышь домовая
# Lepus_europaeus - заяц русак
# Lepus_timidus - заяц беляк
# Canis_lupus - волк
# Cavia_porcellus - морская свинка
# Sus_scrofa - кабан