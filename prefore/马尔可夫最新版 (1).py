import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
file_path = 'C:/Users/2386628107/Desktop/Wimbledon_featured_matches.csv'
data = pd.read_csv(file_path)

# 假设 data 包含了所有必要的列，包括 match_id, set_no, p1_games, p2_games, p1_distance_run, p2_distance_run, speed_mph, point_victor

# 预处理：为了简化，我们假设数据已经清洗好了

# 分析所有比赛
match_ids = data['match_id'].unique()

# 为每场比赛计算转移概率矩阵
for match_id in match_ids:
    match_data = data[data['match_id'] == match_id]
    states = ['P1_win', 'P2_win']
    transition_counts = pd.DataFrame(0, index=states, columns=states)

    for i in range(1, len(match_data)):
        prev_winner = match_data.iloc[i - 1]['point_victor']
        current_winner = match_data.iloc[i]['point_victor']
        transition_counts.loc[states[int(prev_winner)-1], states[int(current_winner)-1]] += 1

    transition_probabilities = transition_counts.div(transition_counts.sum(axis=1), axis=0)
    print(f"Match ID: {match_id}")
    print("转换概率矩阵:")
    print(transition_probabilities)
    print("\n")

    # 可视化：以转换概率矩阵为例，为每场比赛生成一个简单的可视化
    sns.heatmap(transition_probabilities, annot=True, cmap='coolwarm')
    plt.title(f"Transition Probabilities for Match {match_id}")
    plt.show()

import numpy as np
import networkx as nx

def simulate_match(transition_probabilities, num_simulations=1000):
    """
    模拟一场比赛多次，返回P1和P2赢得比赛的概率。
    """
    outcomes = {'P1_win': 0, 'P2_win': 0}
    for _ in range(num_simulations):
        state = 'P1_win'  # 假设比赛从P1获得第一个点开始
        while True:
            next_state = np.random.choice(['P1_win', 'P2_win'], p=transition_probabilities.loc[state])
            if next_state == 'P1_win':
                outcomes['P1_win'] += 1
                break
            elif next_state == 'P2_win':
                outcomes['P2_win'] += 1
                break
            state = next_state
    return outcomes['P1_win'] / num_simulations, outcomes['P2_win'] / num_simulations

# 模拟比赛并计算赢得比赛的概率
p1_win_prob, p2_win_prob = simulate_match(transition_probabilities)
print(f"P1赢得比赛的概率: {p1_win_prob}, P2赢得比赛的概率: {p2_win_prob}")

# 生成与真实数据比较的柱状图
real_p1_win = data[data['point_victor'] == 1].shape[0] / data.shape[0]
real_p2_win = data[data['point_victor'] == 2].shape[0] / data.shape[0]

plt.bar(['P1 Simulation', 'P2 Simulation', 'P1 Real', 'P2 Real'],
        [p1_win_prob, p2_win_prob, real_p1_win, real_p2_win])
plt.ylabel('Win Probability')
plt.title('Simulation vs Real Win Probability')
plt.show()

import networkx as nx
import matplotlib.pyplot as plt  # 确保导入了这个库

# 构建有向图网络
G = nx.DiGraph()

# 添加节点
G.add_node('P1_Win')
G.add_node('P2_Win')

# 添加边和转移概率
G.add_edge('P1_Win', 'P1_Win', weight=0.542642)
G.add_edge('P1_Win', 'P2_Win', weight=0.457358)
G.add_edge('P2_Win', 'P1_Win', weight=0.477005)
G.add_edge('P2_Win', 'P2_Win', weight=0.522995)

# 绘制图
pos = nx.spring_layout(G)  # 为每个节点设置位置
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, edge_color='gray', arrowstyle='->', arrowsize=10)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title('State Transition Graph')
plt.show()


