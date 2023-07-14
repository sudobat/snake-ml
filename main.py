import sys

from agent import Agent
from game import SnakeGameAI
from helper import plot


def play(mode):
    agents = {
        'sneik': Agent(mode, ai_details={
            'name': 'sneik',
            'discount_rate': 0.99,
            'learning_rate': 0.0001,
            'batch_size': 100,
        }),
    }
    game = SnakeGameAI(640, 480, agents)
    while True:
        game.play_step()
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