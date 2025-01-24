# Projeto-Busca-Reinforcement-Learning

Projeto realizado durante o curso da disciplina 'Algoritmos e Estrutura de Dados (2024.2)', no 4° período do curso de Engenharia da Computação na POLI/UPE, mais especificamente no estudo dos Algoritmos de Busca. Trata-se de um mini-game de táxi em que o objetivo é pegar o passageiro em determinado local e levá-lo corretamente a seu destino.

Tendo como base a estrutura do Aprendizado por Reforço (Reinforcement Learning), o algoritmo de busca Q-Learning será o responsável por mapear cada ação do agente no ambiente e enviá-lo um feedback (recompensa), que varia em cada situação. O algoritmo seleciona, a partir da Q-Table montada pelos valores de aprendizado do modelo, a ação que mais favorece o agente (táxi) a cumprir seu objetivo, e assim se segue, por tentativa e erro, até ele conseguir realizar a tarefa consistentemente (aproximadamente 2000 episódios).

# Bibliotecas utilizadas

- Gym: utilizada para simular o ambiente do Reinforcement Learning, que nesse caso foi o 'Taxi-v3'.
- Pandas/NumPy: utilizada para facilitar as operações na Q-Table, que tem um total de 500 estados e 6 ações.
- Pickle: utilizada para manipulação dos arquivos gerados no aprendizado do agente.
- PyGame: utilizada para renderizar a aplicação efetivamente.

# Instalação das bibliotecas e rodando a aplicação

Para instalar as bibliotecas supracitadas e conseguir rodar a aplicação, digite os seguintes comandos no terminal:

1- `pip install numpy gymnasium pandas`\
2- `pip install pygame`\
3- A aplicação estará pronta para ser executada, tanto para visualizar os resultados como para produzir novos.



