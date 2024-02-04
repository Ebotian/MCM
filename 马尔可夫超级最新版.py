import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 加载数据
file_path = '~/MCM/tennis.csv'
data = pd.read_csv(file_path)

# 分析所有比赛
match_ids = data['match_id'].unique()

# 假设data是包含比赛数据的DataFrame，其中包括'match_id', 'server', 和'point_victor'列

# 定义扩展的状态，考虑发球方和得分点的获胜者
extended_states = [
    'P1_serve_win', 'P1_serve_lose',
    'P2_serve_win', 'P2_serve_lose'
]

# 初始化每场比赛的转移概率矩阵
match_transition_probabilities = {}

# 遍历每场比赛
for match_id in data['match_id'].unique():
    match_data = data[data['match_id'] == match_id]

    # 初始化转移计数矩阵
    transition_counts = pd.DataFrame(0, index=extended_states, columns=extended_states)

    # 遍历比赛数据
    for i in range(1, len(match_data)):
        prev_point = match_data.iloc[i-1]
        curr_point = match_data.iloc[i]

        # 确定前一个点和当前点的状态
        prev_state = 'P{}_serve_{}'.format(prev_point['server'], 'win' if prev_point['point_victor'] == prev_point['server'] else 'lose')
        curr_state = 'P{}_serve_{}'.format(curr_point['server'], 'win' if curr_point['point_victor'] == curr_point['server'] else 'lose')

        # 更新转移计数
        transition_counts.loc[prev_state, curr_state] += 1

    # 计算并存储每场比赛的转移概率矩阵
    transition_probabilities = transition_counts.div(transition_counts.sum(axis=1), axis=0)
    match_transition_probabilities[match_id] = transition_probabilities

# 打印每场比赛的转移概率矩阵
for match_id, probabilities in match_transition_probabilities.items():
    print(f"Match ID: {match_id}")
    print("转换概率矩阵:")
    print(probabilities)
    print("\n")
    # 可视化：以转换概率矩阵为例，为每场比赛生成一个简单的可视化
    sns.heatmap(transition_probabilities, annot=True, cmap='coolwarm')
    plt.title(f"Transition Probabilities for Match {match_id}")
    plt.show()
import pandas as  pd
import numpy as np

def simulate_scenario(transition_probabilities, scenario_func, num_simulations=1000):
    """
    通用场景模拟函数，计算并返回选手获胜的概率。
    参数:
    - transition_probabilities: 转移概率矩阵
    - scenario_func: 场景模拟函数，如 simulate_game, simulate_tiebreak, simulate_set 或 simulate_match
    - num_simulations: 模拟次数，默认为1000
    返回:
    - 选手1和选手2获胜的概率
    """
    outcomes = {'P1_win': 0, 'P2_win': 0}
    for _ in range(num_simulations):
        winner = scenario_func(transition_probabilities)
        outcomes[winner] += 1
    p1_win_prob = outcomes['P1_win'] / num_simulations
    p2_win_prob = outcomes['P2_win'] / num_simulations
    return p1_win_prob, p2_win_prob

def simulate_point(transition_probabilities):
    """模拟得分点，返回获胜选手。"""
    return np.random.choice(['P1_win', 'P2_win'], p=[transition_probabilities['P1_win'], transition_probabilities['P2_win']])
# 定义模拟一局、模拟抢七局、模拟一个盘和模拟整场比赛的函数
def simulate_game(transition_probabilities):
    # 模拟一局比赛的逻辑
    score = {'P1_win': 0, 'P2_win': 0}
    while max(score.values()) < 4 or abs(score['P1_win'] - score['P2_win']) < 2:
        point_winner = simulate_point(transition_probabilities)
        score[point_winner] += 1
    return 'P1_win' if score['P1_win'] > score['P2_win'] else 'P2_win'

def simulate_tiebreak(transition_probabilities):
    # 模拟抢七局的逻辑
    score = {'P1_win': 0, 'P2_win': 0}
    while max(score.values()) < 7 or abs(score['P1_win'] - score['P2_win']) < 2:
        point_winner = simulate_point(transition_probabilities)
        score[point_winner] += 1
    return 'P1_win' if score['P1_win'] > score['P2_win'] else 'P2_win'

def simulate_set(transition_probabilities):
    # 模拟一个盘的逻辑
    games_won = {'P1_win': 0, 'P2_win': 0}
    while max(games_won.values()) < 6 or abs(games_won['P1_win'] - games_won['P2_win']) < 2:
        if games_won['P1_win'] == 6 and games_won['P2_win'] == 6:
            tiebreak_winner = simulate_tiebreak(transition_probabilities)
            games_won[tiebreak_winner] += 1
            break
        game_winner = simulate_game(transition_probabilities)
        games_won[game_winner] += 1
    return 'P1_win' if games_won['P1_win'] > games_won['P2_win'] else 'P2_win'

def simulate_match(transition_probabilities, best_of=5):
    # 模拟整场比赛的逻辑
    sets_won = {'P1_win': 0, 'P2_win': 0}
    for _ in range(best_of):
        set_winner = simulate_set(transition_probabilities)
        sets_won[set_winner] += 1
        if sets_won[set_winner] > best_of // 2:
            break
    return 'P1_win' if sets_won['P1_win'] > sets_won['P2_win'] else 'P2_win'

# 为每场比赛计算转移概率矩阵并模拟获胜概率
for match_id in match_ids:
    # 假设已经计算了该场比赛的转移概率矩阵 transition_probabilities
    transition_probabilities = {'P1_win': 0.6, 'P2_win': 0.4}  # 示例转移概率

    # 模拟并输出赢得一局比赛的概率
    p1_win_prob, p2_win_prob = simulate_scenario(transition_probabilities, simulate_game)
    print(f"Match ID {match_id}: 赢得一局比赛的概率 — P1: {p1_win_prob}, P2: {p2_win_prob}")

    # 模拟并输出赢得抢七局的概率
    p1_win_prob, p2_win_prob = simulate_scenario(transition_probabilities, simulate_tiebreak)
    print(f"Match ID {match_id}: 赢得抢七局的概率 — P1: {p1_win_prob}, P2: {p2_win_prob}")

    # 模拟并输出赢得一个盘的概率
    p1_win_prob, p2_win_prob = simulate_scenario(transition_probabilities, simulate_set)
    print(f"Match ID {match_id}: 赢得一个盘的概率 — P1: {p1_win_prob}, P2: {p2_win_prob}")

    # 模拟并输出赢得整场比赛的概率
    p1_win_prob, p2_win_prob = simulate_scenario(transition_probabilities, lambda tp: simulate_match(tp, best_of=5))
    print(f"Match ID {match_id}: 赢得整场比赛的概率 — P1: {p1_win_prob}, P2: {p2_win_prob}")
