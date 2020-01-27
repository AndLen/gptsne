from gptools.moead import MOEAD
from gptsne.eval import evalTSNEMO


class GPTSNEMOEAD(MOEAD):
    DECOMPOSITION = 'tchebycheff'
    obj_mins = [0, 0]
    obj_maxes = [4, 4000]

    def __init__(self, data_t, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_t = data_t
        self.functionType_ = "gpTSNE"

    def updateProblem(self, individual, id_, type_):
        """
        individual : A new candidate individual
        id : index of the subproblem
        type : update solutions in neighbourhood (type = 1) or whole population otherwise.
        """
        time = 0

        if type_ == 1:
            size = len(self.neighbourhood_[id_])
        else:
            size = len(self.population)
        perm = [None] * size

        self.randomPermutations(perm, size)

        for i in range(size):
            k = int()
            if type_ == 1:
                k = self.neighbourhood_[id_][perm[i]]
            else:
                k = perm[i]

            # Don't need to get this again...
            f1 = self.skew_fitness(self.population[k].fitness.values, self.lambda_[k])
            f2 = self.skew_fitness(individual.fitness.values, self.lambda_[k])

            if f2 < f1:  # minimization, JMetal default
                self.population[k] = individual
                time += 1
            if time >= self.nr_:
                self.paretoFront.update(self.population)
                return

    def skew_fitness(self, fitness, lambda_):
        # do this earlier, no?
        fitness *= lambda_
        fitness /= self.obj_maxes

        if self.DECOMPOSITION == 'tchebycheff':
            return fitness.max()
        elif self.DECOMPOSITION == 'weighted':
            return fitness.sum()
        else:
            raise NotImplementedError

    def fitnessFunction(self, individual, lambda_):
        if self.functionType_ != "gpTSNE":
            return super().fitnessFunction(individual, lambda_)
        else:
            fitness = evalTSNEMO(self.data_t, self.toolbox, individual)
            return self.skew_fitness(fitness, lambda_)
