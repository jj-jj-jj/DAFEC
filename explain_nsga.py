import torch
import torch.distributions
import numpy as np
import timeit
import scipy.stats
import tqdm
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from matplotlib import pyplot as plt
import quantus
from deap import base
from deap import creator
from deap import tools
import skimage

print(quantus.helpers.constants.available_methods_captum())

device = torch.device("cuda:0")

CXPB, MUTPB, NGEN = 1, 1, 40 # genetic algorithm params
mode = "squares" # mode for counterfactual generation
eval_model = "efficient" # from - "efficient", "mobilenet", "resnet"
eval_set = "pet" # from - "pet", "food, "stl", "caltech"



def get_loader(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    if eval_set == "pet":
        svhn = datasets.OxfordIIITPet(root = "pet_root", download=True, transform=transform)
    elif eval_set == "food":
        svhn = datasets.Food101(root = "food_root", download=True, transform=transform)
    elif eval_set == "stl":
        svhn = datasets.stl10.STL10(root = "stl_root", download=True, transform=transform)
    else:
        svhn = datasets.caltech.Caltech101(root = "caltech_root", download=True, transform=transform)
    svhn = torch.utils.data.Subset(svhn, np.arange(128))
    svhn_loader = torch.utils.data.DataLoader(dataset=svhn,
                                              batch_size=128,
                                              shuffle=False)
    return svhn_loader


def genetic_test():

    creator.create("FitnessMin", base.Fitness, weights=(1.0, 1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    if mode in ["squares", "circles"]:
        IND_SIZE = 4
    elif mode == "segmentation":
        IND_SIZE = 100
    else:
        IND_SIZE = 8*8

    toolbox = base.Toolbox()
    # toolbox.register("attribute", np.random.binomial, n=1, p=0.01)
    if mode in ["squares", "circles"]:
        toolbox.register("attribute", np.random.randint, low=1, high = 223)
    else:
        toolbox.register("attribute", np.random.binomial, n=1, p=0.5)

    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attribute, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        return sum(individual),

    toolbox.register("mate", tools.cxUniform, indpb=0.1)
    # toolbox.register("mutate", tools.mutUniformInt, low=0, up=1, indpb=0.05)
    if mode == "squares":
        toolbox.register("mutate", tools.mutUniformInt, low=0, up=224, indpb=0.2)
    elif mode == "circles":
        toolbox.register("mutate", tools.mutUniformInt, low=1, up=224, indpb=0.1)
    else:
        toolbox.register("mutate", tools.mutUniformInt, low=0, up=1, indpb=0.05)
    ref_points = tools.uniform_reference_points(2, 50)
    toolbox.register("select", tools.selNSGA3, ref_points=ref_points ,nd='log')
    toolbox.register("evaluate", evaluate)

    return alg(toolbox)
    # plt.imshow(batch[0][0].permute(1,2,0))
    # plt.show()



def alg(toolbox):

    pop = toolbox.population(n=64)

    # Evaluate the entire population
    fitnesses, msizes = eval_on_sample(np.array(pop),seg)
    for ind, fit, msize in zip(pop, fitnesses, msizes):
        ind.fitness.values = [fit, msize]

    for g in range(NGEN):
        # Select the next generation individuals
        nupop = toolbox.select(pop, 32)
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, nupop))
        # Apply crossover and mutation on the offspring
        permute = np.random.permutation(32)
        for child1, child2 in zip([offspring[p] for p in permute[:16]], [offspring[p] for p in permute[16:]]):
            if np.random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.random() < MUTPB:
                toolbox.mutate(mutant)
                # for i in range(4):
                #     if np.random.rand() < 1:
                #         mutant[i] += np.random.randint(-10,11)
                #     if mutant[i]<1:
                #         mutant[i]=1
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses, masksizes = eval_on_sample(np.array(invalid_ind),seg)
        for ind, fit, msize in zip(invalid_ind, fitnesses, masksizes):
            ind.fitness.values = [fit, msize]
        # The population is entirely replaced by the offspring
        pop[:] = nupop + offspring
    fitnesses, masksizes = eval_on_sample(np.array(pop),seg)
    for ind, fit, msize in zip(pop, fitnesses, masksizes):
        ind.fitness.values = [fit, msize]

    return pop

def eval_on_sample(pop, segmentation_mask = None):
    start = timeit.default_timer()
    if mode == "squares":
        xrange = np.arange(x.shape[1])[:, None]
        yrange = np.arange(x.shape[2])[None, :]
        batched_xrange = (xrange>pop[:, 0, None, None]-pop[:, 1, None, None]//4) * (xrange<pop[:, 0, None, None]+pop[:, 1, None, None]//4)
        batched_yrange = (yrange>pop[:, 2, None, None]-pop[:, 3, None, None]//4) * (yrange<pop[:, 2, None, None]+pop[:, 3, None, None]//4)
        masks = 1-batched_xrange*batched_yrange
    elif mode == "circles":
        xrange = np.arange(x.shape[1])[:, None]
        yrange = np.arange(x.shape[2])[None, :]
        batched_xrange = (xrange-pop[:, 0, None, None])**2 / pop[:, 1, None, None]**2
        batched_yrange = (yrange-pop[:, 2, None, None])**2 / pop[:, 3, None, None]**2
        masks = (batched_xrange+batched_yrange)>1
    elif mode == "segmentation":
        pop = pop * np.arange(100)[None]
        pop = (pop[:,:,None,None]==segmentation_mask[None,None])
        masks = pop.sum(axis=1)
    else:
        square = np.reshape(pop, (pop.shape[0], 8, 8))
        mask = np.repeat(square, 28, axis=1)
        masks = 1 - np.repeat(mask, 28, axis=2)
    masked_batch = x * masks[:, None] + x.mean(dim=(1,2),keepdim=True)*(1- masks[:, None])
    out = model(masked_batch.to(device, dtype=torch.float32))
    distance = abs(baseline[:,baseline_label] - out[:,baseline_label])
    sizes = (1 - masks).sum(axis=(1,2))
    return distance.detach().cpu().numpy()/30, (224*224 - sizes)/224/224


def attribution_as_mean(pop, segmentation_mask = None):
    if mode == "squares":
        xrange = np.arange(x.shape[1])[:, None]
        yrange = np.arange(x.shape[2])[None, :]

        batched_xrange = (xrange>pop[:, 0, None, None]-pop[:, 1, None, None]//4) * (xrange<pop[:, 0, None, None]+pop[:, 1, None, None]//4)
        batched_yrange = (yrange>pop[:, 2, None, None]-pop[:, 3, None, None]//4) * (yrange<pop[:, 2, None, None]+pop[:, 3, None, None]//4)
        #
        masks = batched_xrange*batched_yrange

    elif mode == "circles":
        xrange = np.arange(x.shape[1])[:, None]
        yrange = np.arange(x.shape[2])[None, :]
        batched_xrange = (xrange-pop[:, 0, None, None])**2 / pop[:, 1, None, None]**2
        batched_yrange = (yrange-pop[:, 2, None, None])**2 / pop[:, 3, None, None]**2
        masks = (batched_xrange+batched_yrange)<1

    elif mode == "segmentation":
        pop = pop * np.arange(100)[None]
        pop = (pop[:,:,None,None] == segmentation_mask[None,None])
        masks = 1-pop.sum(axis=1)

    else:
        square = np.reshape(pop, (pop.shape[0], 8, 8))
        mask = np.repeat(square, 28, axis=1)
        masks = np.repeat(mask, 28, axis=2)

    return masks.mean(axis=0)


svhn_loader = get_loader(None)

if eval_model == "mobile":
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).to(device)
elif eval_model == "resnet":
    model = models.resnet18(pretrained=True).to(device)
else:
    model = models.efficientnet_b0(pretrained=True).to(device)
model.requires_grad_(True)
model.eval()

for b in svhn_loader:
    batch = b
    break

changes = []
t = tqdm.tqdm(batch[0])
positives = [0,0,0]
evals = 0
metrics = []

metrics.append(
    quantus.FaithfulnessEstimate(
        perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
        similarity_func=quantus.similarity_func.correlation_spearman,
        features_in_step=4*224,
        perturb_baseline="mean",

    )
)
metrics.append(
    quantus.MonotonicityCorrelation(
        nr_samples=1,
        features_in_step=4*224,
        perturb_baseline="mean",
        perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
        similarity_func=quantus.similarity_func.correlation_spearman,
        normalise=True,
        aggregate_func=np.mean,
        return_aggregate=True,
        disable_warnings=True,
    )
)


def try_wilcoxon(wx,wy):
    try:
        return round(scipy.stats.wilcoxon(wx, wy).pvalue, 4)
    except ValueError:
        return 100

wilcoxon = [[[],[]],[[],[]],[[],[]],[[],[]]]

for x in t:

    seg = skimage.segmentation.slic(x.permute((1,2,0)), n_segments=100, start_label=1)

    baseline = model(x[None].to(device))
    baseline_label = baseline.argmax().item()
    start_time = timeit.default_timer()
    pop = genetic_test()
    result = np.array([ind.fitness.values for ind in pop])
    my_time = timeit.default_timer()-start_time
    front = tools.sortLogNondominated(pop, 64, first_front_only=False)[0]

    start_time = timeit.default_timer()
    scores = []

    explanation = attribution_as_mean(np.array(front), seg)[None, None]
    scores.append([
        metric(
            model=model,
            x_batch=x[None].numpy(),
            y_batch=np.array([[baseline_label]]),
            a_batch=explanation,
            device=device
        )
        for metric in metrics
    ])

    explanation = quantus.explain(model, x[None].to(device), baseline_label, method="KernelShap", device=device, xai_lib_kwargs={"feature_mask":torch.tensor(seg, dtype = torch.long, device=device)[None,None]})

    their_time = timeit.default_timer()-start_time
    start_time = timeit.default_timer()
    scores.append([
        metric(
            model=model,
            x_batch=x[None].numpy(),
            y_batch=np.array([[baseline_label]]),
            a_batch=explanation,
            device=device
        )
        for metric in metrics
    ])
    explanation = quantus.explain(model, x[None].to(device), baseline_label, method="GradientShap")

    scores.append([
        metric(
            model=model,
            x_batch=x[None].numpy(),
            y_batch=np.array([[baseline_label]]),
            a_batch=explanation,
            device=device
        )
        for metric in metrics
    ])

    explanation = quantus.explain(model, x[None].to(device), baseline_label, method="IntegratedGradients")

    scores.append([
        metric(
            model=model,
            x_batch=x[None].numpy(),
            y_batch=np.array([[baseline_label]]),
            a_batch=explanation,
            device=device
        )
        for metric in metrics
    ])
    score_time = timeit.default_timer()-start_time


    old_explanation = explanation

    for mtrc in range(0,2):
        for mthd in range(0,4):
            wilcoxon[mthd][mtrc].append(scores[mthd][mtrc][0])

    evals += 1

    out_string = f"Ours: {np.array(wilcoxon[0][0]).mean()} ffllness,  {np.array(wilcoxon[0][1]).mean()} monotonicity"
    for idx, method_name in enumerate(["KernelShap", "GradShap", "IG"]):
        out_string += f"; {method_name}: {np.array(wilcoxon[idx+1][0]).mean()} ffllness with p={try_wilcoxon(wilcoxon[0][0],wilcoxon[idx+1][0])}, {np.array(wilcoxon[idx+1][1]).mean()} monotonicity with p={try_wilcoxon(wilcoxon[idx+1][0],wilcoxon[0][0])}"
    t.set_postfix(positives= out_string)


plt.imshow(attribution_as_mean(np.array(front), seg))
plt.show()

invTrans = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                               transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1, 1, 1 ])])

plt.imshow((np.transpose(invTrans(x),(1,2,0))))
plt.show()

plt.imshow((np.transpose(invTrans(x),(1,2,0)) * attribution_as_mean(np.array(front))[:,:,None]))
plt.show()
# transfer()
