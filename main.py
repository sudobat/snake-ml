import sys

from agent import Agent
from game import SnakeGameAI
from helper import plot


def play(mode):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agents = {
        'model': Agent(mode, 'model'),
        'agent2': Agent(mode, 'agent2'),
    }
    game = SnakeGameAI(640, 480, agents)
    while True:
        # # get old state
        # state_old = agent.get_state(game)
        #
        # # get move
        # final_move = agent.get_action(state_old)

        # perform move and get new state
        game.play_step()
        # state_new = agent.get_state(game)
        #
        # # train short memory
        # agent.train_short_memory(state_old, final_move, reward, state_new, done)
        #
        # # remember
        # agent.remember(state_old, final_move, reward, state_new, done)

        all_lost = True
        for agent in game.agents.values():
            if not agent.game_over:
                all_lost = False
                break

        if all_lost:
            game.reset()
        # if done:
        #     # train long memory, plot result
        #     game.reset()
        #     agent.n_games += 1
        #     agent.train_long_memory()
        #
        #     if score > record:
        #         record = score
        #         agent.model.save(model_name + '.pth')
        #
        #     print('Game', agent.n_games, 'Score', score, 'Record:', record)
        #
        #     plot_scores.append(score)
        #     total_score += score
        #     mean_score = total_score / agent.n_games
        #     plot_mean_scores.append(mean_score)
        #     plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 main.py run | train")
        exit(0)

    play(sys.argv[1])