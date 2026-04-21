import matplotlib.pyplot as plt
import sklearn
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qUpperConfidenceBound
from botorch.optim import optimize_acqf
import warnings

warnings.filterwarnings("ignore", message=".*Minimum element in feature mask is not 0.*")
warnings.filterwarnings("ignore", message=".*It is strongly recommended to simply replace.*")

import torch
import torch.distributions
import numpy as np
import timeit
import scipy.stats
import tqdm
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
device = torch.device("cuda:0")
import quantus
import captum
import sys, io
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective



CXPB, MUTPB, NGEN = 1, 1, 40
dataset_name = "pet" # from [pet, food, caltech101, stl]
model_name = "resnet" # from [resnet, efficientnet, mobilenet]
mode = "square" # from [constant, square, rectangle]
metric_baseline = 'black' # from [mean, black]
square_ratio = 0.3
plot = True
use_softmax = True
images_to_eval = 128

def get_loader(config):
    # mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    if dataset_name == 'pet':
        svhn = datasets.OxfordIIITPet(root = "pet_root", download=True, transform=transform)
    elif dataset_name == 'food':
        svhn = datasets.Food101(root = "food_root", download=True, transform=transform)
    elif dataset_name == 'stl':
        svhn = datasets.stl10.STL10(root = "stl_root", download=True, transform=transform)
    elif dataset_name == 'caltech101':
        svhn = datasets.caltech.Caltech101(root = "caltech_root", download=True, transform=transform)
    else:
        raise ValueError('Dataset not found.')
    svhn = torch.utils.data.Subset(svhn, np.arange(images_to_eval))
    svhn_loader = torch.utils.data.DataLoader(dataset=svhn,
                                              batch_size=images_to_eval,
                                              shuffle=False)
    return svhn_loader



# model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).to(device)



svhn_loader = get_loader(None)
if model_name == "resnet":
    model = models.resnet18(pretrained=True).to(device)
elif model_name == "efficientnet":
    model = models.efficientnet_b0(pretrained=True).to(device)
elif model_name == "mobilenet":
    model = models.mobilenet_v2(pretrained=True).to(device)
else:
    raise ValueError
model.requires_grad_(True)
model.eval()
for b in svhn_loader:
    batch = b
    break

if use_softmax:
    sm = torch.nn.Softmax(dim=1)
    model = torch.nn.Sequential(model,sm)
    model.eval()

changes = []
t = tqdm.tqdm(batch[0])
positives = [0,0,0]
evals = 0
metrics = []
metrics.append(
    quantus.FaithfulnessEstimate(
        perturb_func=quantus.baseline_replacement_by_indices,
        perturb_func_kwargs={"indexed_axes":[1]},
        similarity_func=quantus.correlation_pearson,
        features_in_step=4*224,
        perturb_baseline=metric_baseline,

    )
)

metrics.append(
    quantus.MonotonicityCorrelation(
        nr_samples=1,
        features_in_step=4*224,
        perturb_baseline=metric_baseline,
        perturb_func=quantus.baseline_replacement_by_indices,
        perturb_func_kwargs={"indexed_axes":[1]},
        similarity_func=quantus.correlation_pearson,
        normalise=True,
        aggregate_func=np.mean,
        return_aggregate=True,
        disable_warnings=True,
    )
)

import skimage


def try_wilcoxon(wx,wy):
    try:
        return round(scipy.stats.wilcoxon(wx, wy).pvalue, 4)
    except ValueError:
        return 100

def create_rectangle_mask(corner1, corner2, resolution, angle=0):

    """
        Args:
            corner1: a tensor or np array of shape (batch, 2)
            corner1: a tensor or np array of shape (batch, 2)
            resolution: a tuple (imwidth, imheight)
            angle: unused, leftover from earlier experiments

        Returns:
            a tensor of shape (batch, 1, imwidth, imheight) containing image masks defined by corner position
    """

    corner1 = torch.as_tensor(corner1) # (batch, 2)
    corner2 = torch.as_tensor(corner2) # (batch, 2)
    angle = torch.as_tensor(angle) # (batch, 1)
    batch_size = corner1.shape[0]
    if corner2.shape[0] != batch_size:
        raise ValueError("corner1 and corner2 must have the same batch size")

    if angle.dim() == 0:
        angle = angle.expand(batch_size)
    elif angle.shape[0] != batch_size:
        raise ValueError("angle must be scalar or have the same batch size as corners")

    width, height = resolution

    x = torch.linspace(0, 1, width, device=corner1.device)
    y = torch.linspace(0, 1, height, device=corner1.device)

    xx, yy = torch.meshgrid(x, y, indexing='xy')
    xx = xx.unsqueeze(0).expand(batch_size, -1, -1)  # (B, W, H)
    yy = yy.unsqueeze(0).expand(batch_size, -1, -1)  # (B, W, H)

    x_coords = torch.stack([corner1[:, 0], corner2[:, 0]], dim=1)
    y_coords = torch.stack([corner1[:, 1], corner2[:, 1]], dim=1)

    x_min, _ = torch.min(x_coords, dim=1)
    x_max, _ = torch.max(x_coords, dim=1)
    y_min, _ = torch.min(y_coords, dim=1)
    y_max, _ = torch.max(y_coords, dim=1)

    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    rect_width = torch.clamp(x_max - x_min, min=0.1)
    rect_height = torch.clamp(y_max - y_min, min=0.1)

    center_x = center_x.view(batch_size, 1, 1)
    center_y = center_y.view(batch_size, 1, 1)
    rect_width = rect_width.view(batch_size, 1, 1)
    rect_height = rect_height.view(batch_size, 1, 1)
    angle_rad = torch.deg2rad(angle).view(batch_size, 1, 1)

    xx_centered = xx - center_x
    yy_centered = yy - center_y

    cos_angle = torch.cos(angle_rad)
    sin_angle = torch.sin(angle_rad)

    xx_rot = xx_centered * cos_angle + yy_centered * sin_angle
    yy_rot = -xx_centered * sin_angle + yy_centered * cos_angle

    mask = (torch.abs(xx_rot) <= rect_width / 2) & (torch.abs(yy_rot) <= rect_height / 2)

    return mask[:,None].float()


def eval_on_img(mask):
    """
        Args:
            x: tensor of shape (batch, imwidth, imheight) containing masking values

        Returns:
            change (in response to masking) in the model output for the class originally returned by the model
    """
    baseline = (model(x[None].to(device)))
    baseline_label = baseline.argmax().item()
    masking_value = 0
    if metric_baseline == 'mean':
        masking_value = x.mean(dim=[1,2],keepdim=True)
    out = (model((x[None]*(1-mask) + masking_value*mask).to(device, dtype=torch.float32)))
    distance = baseline[:,baseline_label] - out[:,baseline_label]
    return distance.to(device="cpu")[:,None] #/(0.01 + mask.mean(dim=(2,3))) # + float(baseline_label!=new_label)


def three_arg_black_box_function(x):
    """
        The function to evaluate whether a point is inside a square batch. Required for MOPE optimization loop.

        Args:
            x: tensor of shape (batch, 3) defining square image masks with (location_x, location_y, size).

        Returns:
            evaluation of rectangle masks withing batch
    """
    sqr =  x[:,2:3]
    mask = create_rectangle_mask(x[:,:2]-sqr/2, x[:,:2]+sqr/2, (224, 224))
    return eval_on_img(mask)



class CustomProblemObjective(MCAcquisitionObjective):
    def __init__(self, train_X):
        super(CustomProblemObjective, self).__init__()
        self.train_X = train_X.detach()


    def forward(self, samples, X):
        distance_term = torch.minimum(abs(X[:,:,None]-self.train_X),torch.tensor(0.3)).mean(dim=(2,3))
        return samples.squeeze(-1) + 0*distance_term


class CustomMultiObjective(MCMultiOutputObjective):

    # leftover code from multi-objective botorch experiments, which didn't make it to the paper. Currently only the first objective is used

    def __init__(self, train_X):
        super(CustomMultiObjective, self).__init__()
        self.train_X = train_X.detach()

    def forward(self, samples, X):
        sqr = X[None,:,:,2:3]*0.5+0.1
        second_objective = torch.ones_like(samples)*(1-sqr**2)
        # print(samples.shape, second_objective.shape)
        return torch.cat([samples, second_objective], dim=3)


# the botorch optimization loop for reference BO
def botorch_optimize_loop(bboxfunction, iterations):
    """
        The function to evaluate whether a point is inside a square batch. Required for MOPE optimization loop.

        Args:
            bboxfunction: Black-box function to evaluate.
            iterations: Number of iterations.

        Returns:
            best result
            Gausssian Process regressor for use in calculating the heatmap for Base BO.
    """
    problem_dim = 3
    train_X = torch.rand(16, problem_dim, dtype = torch.double)
    Y = bboxfunction(train_X).to(device="cpu", dtype = torch.double).detach()

    for i in range(iterations):
        gp = SingleTaskGP(
          train_X=train_X,
          train_Y=Y,
          input_transform=Normalize(d=problem_dim),
          outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        bounds = torch.stack([torch.zeros(problem_dim), torch.ones(problem_dim)]).to(torch.double)
        acquisition = qUpperConfidenceBound(model=gp, objective=CustomProblemObjective(train_X=train_X),beta=1)
        lerp_factor = (1+i)/iterations
        candidate, acq_value = optimize_acqf(
          acquisition, bounds=bounds, q=4, num_restarts=5, raw_samples=20, fixed_features=None, timeout_sec
            =0.05
        )
        candidate = torch.rand_like(candidate)
        train_X = torch.cat([train_X, candidate])
        Y = torch.cat([Y,bboxfunction(candidate).to(device="cpu", dtype = torch.double)]).detach()
    gp = sklearn.neighbors.KNeighborsRegressor()
    gp.fit(train_X, Y)
    return train_X[Y.argmax().item()], gp


def point_in_square_batch_from_corners(points: torch.Tensor, squares_min: torch.Tensor, squares_max: torch.Tensor) -> torch.Tensor:
    """
    The function to evaluate whether a point is inside a square batch. Required for MOPE optimization loop.

    Args:
        points: tensor of shape (points, 2)
        squares_min: tensor of shape (squares, 2)
        squares_max: tensor of shape (squares, 2)

    Returns:
        tensor of shape (points,squares)
    """
    pdevice = points.device
    squares_min = squares_min.to(pdevice)
    squares_max = squares_max.to(pdevice)

    points_expanded = points.unsqueeze(1)  # (b, 1, 2)
    min_expanded = squares_min.unsqueeze(0)  # (1, c, 2)
    max_expanded = squares_max.unsqueeze(0)  # (1, c, 2)

    in_square = torch.all(
        (points_expanded >= min_expanded) & (points_expanded <= max_expanded),
        dim=-1
    )  # (b, c)

    return in_square.detach().clone()

def integration_optimize_loop(iterations : int):
    """
        The optimization loop for MOPE

        Args:
            iterations: number of iterations
    """
    point_sample_size = 256
    perturbation_sample_size = 8
    initial_sample_size = 16
    mc_sample = torch.rand([point_sample_size,2])
    mc_sample[0] *=0
    mc_sample[0] +=0.5
    eval_centers = mc_sample[:initial_sample_size]
    eval_sqr_sizes = torch.rand([initial_sample_size,1])*0+0.5
    eval_sqr_sizes[0] *= 0
    eval_sqr_sizes[0] += 1.0

    epsilon = 1e-2
    square_bounds = (torch.clamp(eval_centers-eval_sqr_sizes/2,0,1), torch.clamp(eval_centers+eval_sqr_sizes/2,0,1))
    area = (square_bounds[1][:,:1]-square_bounds[0][:,:1]) * (square_bounds[1][:,1:] - square_bounds[0][:,1:])
    sample_y = three_arg_black_box_function(torch.cat((eval_centers,eval_sqr_sizes),dim=1))/(area+epsilon)   # (points,1) - sampling  squares
    # (b, 1)


    for it in range(iterations):

        mc_sample = torch.rand([point_sample_size, 2])
        candidate_sqr_sizes = torch.rand([point_sample_size, 1])*0.9+0.1

        points_within_bounds = point_in_square_batch_from_corners(mc_sample, eval_centers - eval_sqr_sizes / 2,
                                                           eval_centers + eval_sqr_sizes / 2)
        within_bounds = point_in_square_batch_from_corners(mc_sample, mc_sample - candidate_sqr_sizes / 2,
                                                       mc_sample + candidate_sqr_sizes / 2)

        points_eval = (sample_y.T * points_within_bounds).sum(dim=1, keepdim=True) / points_within_bounds.sum(dim=1,
                                                                                                                  keepdim=True)
        # (points,squares)

        squares_eval = (points_eval*within_bounds).T.sum(dim=1, keepdim=True)/within_bounds.T.sum(dim=1, keepdim=True)
        squares_count_eval = (points_within_bounds.sum(dim=1, keepdim=True)*within_bounds).T.sum(dim=1, keepdim=True)

        nondom_squares_eval = 1-(squares_eval<squares_eval.T).float()
        nondom_squares_std_eval = 1-(squares_count_eval>squares_count_eval.T).float()

        approx_eval =  (nondom_squares_eval*nondom_squares_std_eval).sum(dim=1)

        indices = torch.argsort(approx_eval, descending=True)[:perturbation_sample_size]

        new_candidates = mc_sample[indices]
        new_eval_sqr_sizes = candidate_sqr_sizes[indices]
        square_bounds = (torch.clamp(new_candidates-new_eval_sqr_sizes/2,0,1), torch.clamp(new_candidates+new_eval_sqr_sizes/2,0,1))
        area = (square_bounds[1][:,:1]-square_bounds[0][:,:1]) * (square_bounds[1][:,1:] - square_bounds[0][:,1:])
        new_sample_y = three_arg_black_box_function(torch.cat((new_candidates,new_eval_sqr_sizes),dim=1))/(area+epsilon)

        eval_centers = torch.cat((eval_centers,new_candidates),dim=0).detach()
        eval_sqr_sizes = torch.cat((eval_sqr_sizes,new_eval_sqr_sizes),dim=0).detach()
        sample_y = torch.cat((sample_y,new_sample_y),dim=0).detach()

    return (eval_centers,eval_sqr_sizes,sample_y), None

def optimize():

    # optimize function to get a heatmap from BO results

    black_box_function = three_arg_black_box_function

    out, gp_regressor = botorch_optimize_loop(black_box_function, 16)

    X = np.random.rand(512,3)
    y = gp_regressor.predict(X)[:,0]
    sqr =  X[:,2:3]*square_ratio+square_ratio/2
    mask = create_rectangle_mask(X[:,:2]-sqr/2, X[:,:2]+sqr/2, (224, 224))
    heatmap = ((mask*y[:,None,None,None])/mask.sum(dim=(2,3),keepdims=True)).sum(dim=(0,1))
    heatmap += eval_on_img(torch.ones((224,224)))/224/224
    mask_frequency = mask.sum(dim=(0,1)) + 1
    heatmap = heatmap/(mask_frequency+1e-5)

    return heatmap[None,None].detach().numpy()

def optimize_integrate():

    # optimize function to get a heatmap from MOPE

    (eval_centers,eval_sqr_sizes,sample_y), gp_regressor = integration_optimize_loop(8)
    heatmap = []
    for i in range(224):
        in_tensor = torch.zeros((224,2))
        in_tensor[:,0] += i/224
        in_tensor[:,1] += torch.arange(0,1,1/224)
        points_within_bounds = point_in_square_batch_from_corners(in_tensor, eval_centers - eval_sqr_sizes / 2,
                                                           eval_centers + eval_sqr_sizes / 2)
        points_eval = (sample_y.T * points_within_bounds).sum(dim=1, keepdim=True) / points_within_bounds.sum(dim=1,
                                                                                                                  keepdim=True)
        heatmap.append(points_eval)
    heatmap_tensor = torch.cat(heatmap,dim=1).detach()
    if (heatmap_tensor<=0).all():
        heatmap_tensor = torch.rand((224,224))
    return heatmap_tensor[None,None].detach().cpu().numpy(), create_rectangle_mask(eval_centers-eval_sqr_sizes/2, eval_centers+eval_sqr_sizes/2, (224, 224))

if plot:
    rowlabels = ["Image", "Saliency", "Sampling"]
    plt.ion()
    fig, axes = plt.subplots(3, 5, figsize=(10, 4))
    fig.tight_layout(pad=1.0)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    # Add some sample data to each subplot
    for i, ax in enumerate(axes.flat):
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel(f' {rowlabels[i // 5]}', fontsize=12)


wilcoxon = np.zeros((2,4,images_to_eval))
plotidx = 0
for x in t:
    # try:
        plotidx += 1
        times = []
        seg = skimage.segmentation.slic(x.permute((1,2,0)), n_segments=100, start_label=1, compactness=0.01)
        # seg = None
        baseline = model(x[None].to(device))
        baseline_label = baseline.argmax()
        start_time = timeit.default_timer()

        explanation1  = optimize()
        times.append(timeit.default_timer()-start_time)

        scores1 = [
            metric(
                model=model,
                x_batch=x[None].cpu().numpy(),
                y_batch=np.array([baseline_label.cpu()]),
                a_batch=explanation1,
                # channel_first=True,
                softmax=use_softmax,
                device=device
            )
            for metric in metrics
        ]

        start_time = timeit.default_timer()
        explanation2 = captum.attr.KernelShap(model).attribute(x[None].to(device), baselines=None, n_samples=80, target=baseline_label, feature_mask = torch.tensor(seg, dtype = torch.long, device=device)[None,None])
        times.append(timeit.default_timer()-start_time)

        scores2 = [
            metric(
                model=model,
                x_batch=x[None].cpu().numpy(),
                y_batch=np.array([baseline_label.cpu()]),
                a_batch=explanation2.cpu().numpy(),
                softmax=use_softmax,
                device=device
            )
            for metric in metrics
        ]
        start_time = timeit.default_timer()
        explanation3 = quantus.explain(model, x[None].to(device), baseline_label, xai_lib="captum", method="IntegratedGradients", device=device)
        times.append(timeit.default_timer()-start_time)

        scores3 = [
            metric(
                model=model,
                x_batch=x[None].cpu().numpy(),
                y_batch=np.array([baseline_label.cpu()]),
                a_batch=explanation3,
                softmax=use_softmax,
                device=device
            )
            for metric in metrics
        ]


        start_time = timeit.default_timer()
        explanation, sampling = optimize_integrate()
        times.append(timeit.default_timer()-start_time)

        if plot:
            inv_mean = torch.tensor([0.485, 0.456, 0.406])
            inv_std = torch.tensor([0.229, 0.224, 0.225])
            axes[0,plotidx%5].imshow(x.permute((1,2,0))*inv_std+inv_mean)
            axes[1,plotidx%5].imshow(explanation[0,0])
            axes[2,plotidx%5].imshow(sampling.mean(dim=(0,1)),cmap='inferno')



        scores_ours = [
            metric(
                model=model,
                x_batch=x[None].cpu().numpy(),
                y_batch=np.array([baseline_label.cpu()]),
                a_batch=explanation,
                softmax=use_softmax,
                device=device
            )
            for metric in metrics
        ]
        for sc in range(0,2):
            wilcoxon[sc,0,t.n]=scores_ours[sc][0]
            wilcoxon[sc,1,t.n]=scores1[sc][0]
            wilcoxon[sc,2,t.n]=scores2[sc][0]
            wilcoxon[sc,3,t.n]=scores3[sc][0]
            # wilcoxon_z[sc].append(scores2[sc][0])
            # wilcoxon_zz[sc].append(scores3[sc][0])

        evals += 1
        t.set_postfix(results= f"data: {dataset_name}, model: {model_name}, mode: {mode},  "
                                 f"met 1: {[ '%.2f' % wilcoxon[0,i,:t.n+1].mean() for i in range(4)]}, "
                                 f"met 2: {['%.2f' % wilcoxon[1,i,:t.n+1].mean() for i in range(4)]}, "
                                 f"wcx 1: {[try_wilcoxon(wilcoxon[0,0,:t.n+1],wilcoxon[0,i,:t.n+1]) for i in range(1,4)]}, "
                                 f"wcx 2: {[try_wilcoxon(wilcoxon[1,0,:t.n+1],wilcoxon[1,i,:t.n+1]) for i in range(1,4)]}, "
                      f"times: {['%.3f' % tim for tim in times]}")

        # print(f"data: {dataset_name}, model: {model_name}, mode: {mode},  "
        #                          f"mean values for metric 1: {np.array(wilcoxon_x[0]).mean(),np.array(wilcoxon_y[0]).mean(),np.array(wilcoxon_z[0]).mean(),np.array(wilcoxon_zz[0]).mean()}, "
        #                          f"mean valies for metric 2: {np.array(wilcoxon_x[1]).mean(),np.array(wilcoxon_y[1]).mean(),np.array(wilcoxon_z[1]).mean(),np.array(wilcoxon_zz[1]).mean()}, "
        #                          f"wilcoxon result: {[try_wilcoxon(wx,wy) for wx,wy in zip(wilcoxon_x,wilcoxon_y)],[try_wilcoxon(wx,wy) for wx,wy in zip(wilcoxon_x,wilcoxon_z)],[try_wilcoxon(wx,wy) for wx,wy in zip(wilcoxon_x,wilcoxon_zz)]}, "
        #                          f" times {my_time:.2f}, {their_time:.2f}, {score_time:.2f}")
        if plot:
            fig.canvas.draw_idle()
            plt.pause(0.1)
        # np.save(f'results/{dataset_name}_{model_name}_{metric_baseline}.npy', wilcoxon)
    # except Exception:
    #     print("shite")
    #     pass

plt.show()
