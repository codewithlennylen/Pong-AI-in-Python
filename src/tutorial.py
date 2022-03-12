from pong import Game
import pygame
import neat
import os
import pickle

width, height = 700,500
window = pygame.display.set_mode((width,height))



class PongGame:
    def __init__(self, window, window_width, window_height):
        self.game = Game(window, window_width, window_height)
        self.left_paddle = self.game.left_paddle
        self.right_paddle = self.game.right_paddle
        self.ball = self.game.ball
    
    def test_ai(self, genome, config):
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        run = True
        clock = pygame.time.Clock()
        while run:
            clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                self.game.move_paddle(left=True, up=True)
            if keys[pygame.K_DOWN]:
                self.game.move_paddle(left=True, up=False)

            
            output = net.activate((self.right_paddle.y,self.ball.y, abs(self.right_paddle.x-self.ball.x)))
            decision = output.index(max(output))

            if decision == 0:
                # stay still
                # self.game.move_paddle()
                pass
            elif decision == 1:
                # move up
                self.game.move_paddle(left=False, up=True)
            elif decision == 2:
                # move down
                self.game.move_paddle(left=False, up=False)


            game_info = self.game.loop()
            print(game_info.left_score, game_info.right_score)
            self.game.draw()
            pygame.display.update()

        pygame.quit()

    def train_ai(self, genome1, genome2, config):
        net1 = neat.nn.FeedForwardNetwork.create(genome1,config)
        net2 = neat.nn.FeedForwardNetwork.create(genome2,config)

        run = True
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

            # The outputs are numeric values related to each of our three outputs
            output1 = net1.activate((self.left_paddle.y,self.ball.y, abs(self.left_paddle.x-self.ball.x)))
            decision1 = output1.index(max(output1))

            if decision1 == 0:
                # stay still
                # self.game.move_paddle()
                pass
            elif decision1 == 1:
                # move up
                self.game.move_paddle(left=True, up=True)
            elif decision1 == 2:
                # move down
                self.game.move_paddle(left=True, up=False)


            output2 = net2.activate((self.right_paddle.y,self.ball.y, abs(self.right_paddle.x-self.ball.x)))
            decision2 = output2.index(max(output2))

            if decision2 == 0:
                # stay still
                # self.game.move_paddle()
                pass
            elif decision2 == 1:
                # move up
                self.game.move_paddle(left=False, up=True)
            elif decision2 == 2:
                # move down
                self.game.move_paddle(left=False, up=False)

            # print(output1, output2)

            game_info = self.game.loop()
            self.game.draw(draw_score=False, draw_hits=True)
            pygame.display.update()

            if game_info.left_score >= 1 or game_info.right_score >= 1 or game_info.left_hits > 50:
                self.calculate_fitness(genome1, genome2, game_info)
                break
    
    def calculate_fitness(self, genome1, genome2, game_info):
        genome1.fitness += game_info.left_hits
        genome2.fitness += game_info.right_hits

def eval_genomes(genomes,config):
    width, height = 700,500
    window = pygame.display.set_mode((width, height))

    for i, (genome1_id, genome1) in enumerate(genomes):
        if i == len(genomes) - 1:
            break
        genome1.fitness = 0
        for genome2_id, genome2 in genomes[i+1:]:
            genome2.fitness = 0 if not genome2.fitness else genome2.fitness
            game = PongGame(window, width, height)
            
            game.train_ai(genome1, genome2, config)

def run_neat(config):
    # You also have the ability to restart training from a certain checkpoint.
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-6')
    # p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    # pass a fitness_function, max_no_of_generations
    winner = p.run(eval_genomes, 50)

    with open("best-nn.pickle","wb") as f:
        pickle.dump(winner, f)

def test_ai(config):
    with open("best-nn.pickle","rb") as f:
        winner = pickle.load(f)

    width, height = 700,500
    window = pygame.display.set_mode((width, height))

    game = PongGame(window, width, height)
    game.test_ai(winner, config)



if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_dir = os.path.join(local_dir, 'config.txt')

    config = neat.Config(
                         neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet,neat.DefaultStagnation,
                         config_dir
                         )

    # run_neat(config)

    test_ai(config)
    