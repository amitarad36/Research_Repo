# Speaker notes — reviewed opening

Three spoken sentences per slide. These notes follow slide titles rather than PDF page numbers, because section dividers change numbering. Later theory and research slides still need review.

## Title
I am studying how stop-gradient changes learning dynamics, especially in self-supervised learning. I will move from small examples to PEIRA and then discuss possible research directions. The aim is to identify concrete mathematical questions and discuss their usefulness and feasibility.

## Outline
We begin with what the operator changes in backpropagation. Then we connect those changes to self-supervised learning and the problem of representation collapse. Finally, we examine PEIRA and possible extensions of its analysis.

## One line of code, everywhere
Stop-gradient returns the same value in the forward pass but assigns zero derivative to that path during backpropagation. This holds the value fixed for differentiation, not across training steps. The operation appears in several learning methods, and we will focus on self-supervised learning.

## One dimension: the loss loses its authority
Numerically, this loss is minus one-half x squared, which decreases as the magnitude of x grows. But the detached derivative is zero everywhere, so the training update never moves. The update field has a constant potential, which is different from the displayed forward loss.

## Two dimensions: the update has no potential
The forward loss is identically zero, yet the update field is nonzero away from the origin. In continuous time, squared radius is conserved and the trajectories circle, while ordinary simultaneous discrete steps increase the radius. The unequal mixed derivatives show that this field has no scalar potential.

## A game interpretation of this example
The x update follows the objective xy, while the y update follows the opposing objective minus xy. Each derivative holds the other variable fixed, but both variables move during training. Individual descent directions can therefore produce joint cycles.

## Stopping, improving, and staying nearby
A fixed point means all updates vanish, while a Nash equilibrium means no player can lower its own objective by changing alone. In our circular example, the origin is both: each player's objective is constant when the other variable is zero. Nearby trajectories stay on small circles, so the origin is stable but does not attract them.
## Finding a Lyapunov function
Even when the update has no potential, we can search for another scalar quantity that decreases along its trajectories. Strict decrease away from equilibria rules out cycles, because returning to the same state would require returning to the same value. A possible research goal is to find such a certificate for a specific stop-gradient system, for example by testing whether PEIRA's certificate extends to a predictor that learns at finite speed.
## Learning without labels still needs a signal
Self-supervised learning constructs a learning signal from the data rather than human labels. Here we create two augmented views and encode each of them. We want representations that retain useful information shared between the views despite their differences.

## Agreement alone admits a useless solution
Matching the representations of two views sounds like a natural learning objective. But a constant encoder achieves perfect agreement while distinguishing none of the inputs. We therefore need to understand not only whether training converges, but whether it avoids this collapsed solution.

# Details to remember during discussion

- For equilibrium stability, a classical Lyapunov function also needs a strict local minimum at the equilibrium; nonincrease alone is not the full stability theorem.
- Under standard smoothness assumptions and a compact positively invariant region, LaSalle's principle gives approach to the largest invariant subset of the zero-derivative set, not automatically convergence to a single point.
- A conserved radius bounds our circular example but does not dissipate anything.
- In the scalar shared-target model, define w, p, k, and s before introducing beta: w is the encoder weight, p the predictor weight, k the online-view second moment, and s the cross-view second moment.
- That scalar model is an idealization, not the full BYOL system with an EMA teacher; independent modes require additional structural assumptions.
- For fixed k and s, beta = s - pk gives beta-dot = -k w^2 beta and the derivative of beta squared is -2 k w^2 beta squared.
- This derivative can vanish at w = 0 even when beta is nonzero, so decreasing beta squared does not by itself prove predictor optimality or avoidance of collapse.

- Game qualification: players control distinct parameter blocks, although objectives can depend on all blocks; in smooth unconstrained games Nash implies a fixed point, but the converse need not hold.



## Self-distillation: predict another view
We make two augmented views of one image and pass them through an online branch and a target branch. The online branch includes a predictor whose output is trained to match the target representation, with gradients blocked through the target. The next distinction is how the target weights are updated between training steps.

## SimSiam and BYOL: where does the target come from?
SimSiam uses the same current encoder weights for both views and detaches the target output in each directed loss. BYOL instead has a separate target encoder whose weights slowly track the online encoder through an exponential moving average. Both have an online predictor and train in both view directions, so neither is just naive representation matching.

## From a network to a scalar model
A linear encoder maps a d-dimensional input to an n-dimensional representation, and a predictor maps that representation to another vector of the same size. To make the mechanism transparent, we choose both dimensions to be one, so the encoder and predictor are just real numbers w and p. This is a shared-target idealization with equal learning rates, no weight decay, and no moving-average teacher, rather than a theorem about the full architecture.

## The scalar prediction task
The online branch predicts pw times the first view, while the target is w times the second view with its gradient blocked. We measure their squared difference averaged over the data. The constants k and s record the views' second moments and cross-view correlation, while only w and p are trained.

## Deriving the predictor update
Differentiating squared error gives the residual times the derivative of the prediction, which here is w times the first view. Averaging produces w squared times pk minus s, and gradient flow reverses that sign. We call s minus pk beta, so the predictor moves toward s divided by k whenever the encoder is nonzero.

## Deriving the encoder update: detach matters
For the encoder derivative, the stopped target contributes no derivative even though it numerically contains the same w. The residual is therefore multiplied only by p times the first view. This gives w-dot equal to pw beta; removing detach would also differentiate the target and would change that equation.

## Zero loss does not mean attraction
At a collapsed encoder, both prediction and target vanish, so the nonnegative squared-error loss reaches its global minimum of zero. Near this point, the predictor changes only at second order in the small encoder weight, while the encoder has linear growth rate p-zero times s minus p-zero k. If this rate is positive, a small perturbation grows away from collapse: a loss minimum can repel under the detached update.
## Finding a decreasing quantity
We care about a decreasing quantity because it tracks progress even when the displayed loss does not decrease, and strict decrease away from equilibria would rule out cycles. Here, differentiating beta squared gives minus two k times w squared times beta squared, so the predictor mismatch cannot increase. This is a step toward convergence analysis, but decrease can stop at either zero mismatch or collapse, so it does not yet tell us that training reaches a useful representation.
## Training can stop for two different reasons
The predictor mismatch stops decreasing when either the predictor reaches its preferred scale or the encoder becomes zero. A zero encoder makes prediction and target vanish, shutting down every update even if the predictor scale is wrong. The decreasing quantity alone therefore cannot distinguish a matched predictor from representation collapse, and a matched predictor does not itself establish representation quality.

## An invariant can rule out collapse in the toy model
The equations keep p squared minus w squared fixed at its initial value, which restricts reachable states. Starting from p equal to zero and w equal to one forces w squared to remain at least one, so collapse is impossible for this initialization in this model. This illustrates what an invariant can establish, but richer representations have interacting directions and partial collapse, motivating the move beyond the scalar example.

## From the toy example to PEIRA
Our toy model showed how detach changes learning dynamics, but it used scalar weights. We now turn to a paper that studies richer representations. We want to understand whether training converges, what it learns, and whether collapse is stable.
## The baseline: two encoders and a predictor
The paired inputs x and y are mapped by nonlinear encoders U and V into k-dimensional vectors, and a shared linear predictor P maps between their representations. Each encoder predicts the other view, with gradients passing through its prediction branch but blocked through its target; both encoders therefore learn. Whenever the encoders change, the predictor is assumed to reach the exact minimum of its regularized regression problem, which includes a penalty on large predictor weights.
## CCA: finding what two views share
Imagine many images with paired augmented views, and search for one feature of each view whose outputs vary together strongly. Normalize their scale, then look for further pairs orthogonal to the earlier ones so they do not simply repeat the same feature. Nonlinear CCA searches over square-integrable functions and organizes shared structure into ordered feature pairs, each computed from its own view.

## How does PEIRA use CCA?
In canonical coordinates, each paired direction has a correlation strength, while different pairs have zero cross-correlation. The authors use this structure to express encoder equilibria and analyze whether perturbations in missing directions grow or shrink. CCA therefore describes what training can retain, rather than being a separate preprocessing algorithm executed before training.
## From analyzing self-distillation to changing the objective
CCA provides shared directions for describing equilibria and analyzing stability, while a separately constructed Lyapunov certificate helps establish convergence. The original self-distillation model can nevertheless converge to stable zero or partially collapsed representations. This motivates PEIRA's new objective, whose stable equilibria retain the leading eligible canonical directions under the paper's assumptions.
## PEIRA: a new objective with better stable outcomes
PEIRA minimizes a negative reward for the trace of the best regularized cross-view predictor plus a penalty on representation size, rather than minimizing prediction error. Under the paper's function-space and spectral assumptions, its stable equilibria are global minimizers retaining the strongest eligible canonical directions up to the representation capacity. Regularization sets eligibility, and informative nonconstant directions must exist; missing an eligible direction makes an equilibrium unstable, unlike the stable deficient outcomes of the original self-distillation model.
## Research questions: from one model to a general method
We can enter the problem by letting PEIRA's predictor learn at finite speed and asking whether its self-distillation certificate extends. From that example, we want to identify a constructive recipe using residuals and tracking errors, and structural conditions under which the recipe transfers to other stop-gradient models. Beyond convergence, we must separately investigate collapse stability and the shared features retained, rather than assume a decreasing certificate answers those questions.
Source for PEIRA slides: https://arxiv.org/html/2605.17671v1, Sections 3–5, Propositions 4.1–4.4 and footnote 2.

## PEIRA technical reference — not for reading aloud
With U and V denoting paired encoded views, N = E[UU^T + VV^T] and Sigma = E[VU^T + UV^T]. The exact predictor is P* = Sigma (N + lambda I)^(-1). The self-distillation certificate is Tr[(N + lambda I)^3]/3 - Tr[Sigma^2]/2. The PEIRA objective is -Tr(P*)/2 + lambda Tr(N)/2. Eligible canonical modes satisfy c_i > lambda^2. The convergence and stability statements rely on the compactness/spectral conditions and distinct nonzero canonical correlations used in the paper; exact initialization at an unstable equilibrium need not escape it.



CCA discussion detail: the paper's uncentered function-space formulation includes the constant pair with correlation one; informative variation requires distinguishing nonconstant directions. Canonical functions are orthonormal within each view, and the cross-correlated subspace is complemented by directions with no cross-view signal. This spectral decomposition relies on the paper's assumptions.






