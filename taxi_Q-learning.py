import numpy as np
import gymnasium as gym
import pickle
import random
import pandas as pd

def treinarAgente(numeroEpisodios):

    amb = gym.make('Taxi-v3', render_mode='human')
    tabelaQ = np.zeros((amb.observation_space.n, amb.action_space.n)) # tabela Q 500 x 6

    # fatores de aprendizado/correção
    alfa = 0.5
    gama = 0.9
    epsilon = 1.0

    for episodio in range(numeroEpisodios):

        estado = amb.reset()[0]
        terminou = False
        truncado = False

        totalRecompensa = 0

        while not terminou and not truncado:

            if random.uniform(0, 1) < epsilon:
                acao = amb.action_space.sample()  # exploração total

            else:
                acao = np.argmax(tabelaQ[estado])  # ação pensada de acordo com os valores da tabela Q
            
            proximoEstado, recompensa, terminou, truncado, info = amb.step(acao) # gerar o próximo estado e as informações sobre ele
            qAntigo = tabelaQ[estado, acao]
        
            proximoMaximo = np.max(tabelaQ[proximoEstado]) # escolha do maior valor para o novo estado na tabela Q
            tabelaQ[estado, acao] = (1 - alfa) * qAntigo + alfa * (recompensa + gama * proximoMaximo) # função Q em ação

            totalRecompensa += recompensa
            estado = proximoEstado
        
        epsilon = max(0.1, epsilon * 0.995) # decrescendo o fator de exploração aos poucos e focando na aplicação via tabela Q

    print("Treinamento finalizado no episódio {}, com {} pontos de recompensa.".format(episodio, totalRecompensa))
    amb.close()

def visualizarAprendizadoAgente(episodiosTreinados):

    tabelaQ = carregarTabelaQ(episodiosTreinados)

    amb = gym.make('Taxi-v3', render_mode='human')

    estado = amb.reset()[0] 

    terminou = False
    truncado = False

    totalRecompensa = 0

    while not terminou and not truncado:

        acao = np.argmax(tabelaQ[estado])
        proximoEstado, recompensa, terminou, truncado, info = amb.step(acao)

        totalRecompensa += recompensa
        estado = proximoEstado

        amb.render()

        print(" Estado: {}, Ação: {}, Recompensa: {}".format(estado, acao, recompensa))

    print("Total de recompensa: {}".format(totalRecompensa))
    amb.close()

# carregando um estado do agente treinado com um certo número de episódios
def carregarTabelaQ(episodiosTreinados):

    if episodiosTreinados == 500:
        with open('tabelaQ_taxi_500.pkl', 'rb') as f:
            return pickle.load(f)
        
    if episodiosTreinados == 2000:
        with open('tabelaQ_taxi_2000.pkl', 'rb') as f:
            return pickle.load(f)
        
    elif episodiosTreinados == 10000:
        with open('tabelaQ_taxi_10000.pkl', 'rb') as f:
            return pickle.load(f)
    

def salvarTabelaQ(tabelaQ, filename="tabelaQ_10.csv"):

    df = pd.DataFrame(tabelaQ)
    df.to_csv(filename, index_label="Estado",header=[" Sul ", " Norte ", " Leste ", " Oeste ", " Pegar Passageiro ", " Deixar Passageiro "])

    with open("tabelaQ_taxi_10.pkl", "wb") as f:
        pickle.dump(tabelaQ, f)


def main():

    # pode-se escolher entre treinar o agente até um determinado número de episódios ou visualizar seu aprendizado:

    # treinarAgente(10)
    visualizarAprendizadoAgente(2000) # 2000 episódios é um número de confiança p/ o agente cumprir a tarefa de levar o passageiro corretamente

main()