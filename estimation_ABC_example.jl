import Distributed: @everywhere  # Sera utile pour la parallélisation
# @everywhere sera à mettre devant toute fonction et variable

# Ligne suivante peut-être pas nécessaire avec la nouvelle façon d'installer le package de Mahmoud
@everywhere push!(LOAD_PATH, "/home/gurvan/Documents/divers/markovProcesses/markovprocesses.jl/core")
@everywhere using MarkovProcesses # Statistical framework to perform SMC-ABC
# Available at:
# https://gitlab-research.centralesupelec.fr/2017bentrioum/markovprocesses.j

@everywhere using Distributions
using Plots

include("/home/chirraneso/genealogy_abc_script/phylogeny_simu_abc.jl")
#include("C:\\Users\\sophi\\phylogenetic_tree\\code_arbre\\stage\\genealogy_abc_script\\simu_real_data.jl")

# L'objet suivant - de type Chemical Reaction Network - ne sera pas directement utilisé
# Juste un moyen pratique et rapide pour définir des paramètres, en contournant l'usage initial prévu par le package
dummy = @network_model begin
    sym: (HSC => 2HSC, p2*HSC) #Symetrical divisions
    asym: (HSC => 1HSC + abs, (1.0-p2-p0)*HSC) #asymetrical divisions. NB: abs would correspond to a progenitor cell, but actually we do not model progenitor cells here
    diff: (HSC => 0, p0*HSC(1.0 + lambda -lambda)) #differentiated divisions and dummy way to introduce other parameters
    end "Process for mutated stem cells"

# La vrai structure du modèle est la suivante
@everywhere mutable struct ComplexModel <: Model
    network_model::ContinuousTimeModel
    # Puis on ajoute différents éléments qui servent à définir le modèle
    t_max::Int64
    nb_leafs::Int64
end

# Instanciation de ce modèle :
modelMPN = ComplexModel(dummy, 
                    800, #t_max
                    22 #nb_leafs
)

# Ce qui contiendra l'output du modèle (simu), pour comparer aux observations
@everywhere struct ComplexTrajectory <: AbstractTrajectory
    tree::PhyloTree
end

@everywhere import MarkovProcesses: get_proba_model
@everywhere get_proba_model(model::ComplexModel) = model.network_model

#@everywhere lambda = 1 # Plus tard, un paramètre à estimer

#Il faut découpler la partie simumation de modèle de la partie estimation

#Simulation
@everywhere function simulate(model::ComplexModel; 
    p::Union{Nothing,AbstractVector{Float64}} = nothing)

#On récupère les paramètres dont les noms ont été définis dans dummy
p_sim = (model.network_model).p
param_to_idx = model.network_model.map_param_idx

p0 = p_sim[param_to_idx[:p0]]
p2 = p_sim[param_to_idx[:p2]]
lambda = p_sim[param_to_idx[:lambda]]

t_max = model.t_max
nb_leafs = model.nb_leafs

##### @Sophia : Je copie le début de ta fonction "compute_distance" 

result=simu_complete(lambda,p0,p2,nb_leafs,t_max)

#println(!isempty(findall(result[1].==nothing)))
v_time=result[1]
v_ancestor=result[2]
count_iter=result[3]

edge_list,root = to_edge_list(v_time,v_ancestor,t_max,lambda)
edge_root = (root,"STM_00",0.00) # add a wild type stem cell for vizualisation
phylogeny_simu= create_phylotree(edge_list,edge_root)

#####

return ComplexTrajectory(phylogeny_simu)
end


# On peut tester la fonction simulate
# En choisissant par exemple les vraies valeurs utilisées pour générer l'observation
set_param!(modelMPN, [0.24, 0.2, 1] #p2, p0, l'ordre correspond à celui dans lequel on a définit les params dans dummy
            );

simulate(modelMPN)


# Distance function for hybrid_invasion model
@everywhere function hybrid_dist_obs(vec_sim, vec_observations) 
    #vec_sim is a vector containing Np different trajectories
    #vec_observations is the vector of our Np observations 

    #On choisit pour le moment Np = 1
    mean_dist = 0
    for i in 1:100

        phylogeny_simu = vec_sim[i].tree #Grosso modo, le retour de la fonction simulate
        
        Y_ref = vec_observations[i] #Selon comment on a défini vec_observations[1]
        
        ##### @Sophia : Je copie la 2e partie de ta fonction "compute_distance" 
        dw1,paths1=tree2dyckword_step(phylogeny_simu.root.childs[1][1].childs[1][1],Dict([]),0)
        sort_dw1=triFusion_dw_step(phylogeny_simu.root.childs[1][1].childs[1][1],dw1,paths1,0)
        Y1=dyck_word2dyck_path(sort_dw1)

        mean_dist +=  minkowski(Y1,Y_ref,1)
    end
    
    return mean_dist/100

    
    #####
end
Y_ref=[0.0, 1.9462890625, 13.09649367559524, 117.92515134604974, 139.15757322104974, 141.95634216044368, 178.96093774867904, 189.02512693786832, 558.9696916852679, 558.9696916852679, 928.9142564326675, 928.9142564326675, 928.9142564326675, 1308.9230103692562, 1308.9230103692562, 1308.9230103692562, 1725.9363598940804, 1725.9363598940804, 1725.9363598940804, 1733.5125909546864, 2145.7484783582986, 2145.7484783582986, 2557.984365761911, 2557.984365761911, 2557.984365761911, 2557.984365761911, 2599.548818886911, 2665.2693049980217, 2711.038535767252, 2999.0289061011285, 2999.0289061011285, 3287.019276435005, 3287.019276435005, 3287.019276435005, 3620.778877538112, 3620.778877538112, 3620.778877538112, 4020.25896475233, 4020.25896475233, 4020.25896475233, 4020.25896475233, 4056.1568695818755, 4566.132162762003, 4566.132162762003, 5076.1074559421295, 5076.1074559421295, 5076.1074559421295, 5076.1074559421295, 5133.660376329034, 5633.130858564897, 5633.130858564897, 6132.60134080076, 6132.60134080076, 6132.60134080076, 6132.60134080076, 6250.90309861326, 6283.10903611326, 6297.574781305568, 6298.758114638901, 6691.571032486028, 6691.571032486028, 7084.383950333156, 7084.383950333156, 7084.383950333156, 7478.380201513616, 7478.380201513616, 7478.380201513616, 7487.9144563213085, 7886.842197886384, 7886.842197886384, 8285.76993945146, 8285.76993945146, 8285.76993945146, 8285.76993945146, 8321.43900195146, 8325.479511210719, 8726.437873324227, 8726.437873324227, 9127.396235437735, 9127.396235437735, 9127.396235437735, 9532.395106810502, 9532.395106810502, 9532.395106810502, 9532.395106810502]
vec_observations = [(Y_ref) for k in 1:100]

@time epsilon = hybrid_dist_obs([simulate(modelMPN) for k in 1:100], vec_observations)
@show epsilon #Donne une idée de la distance avec le choix de paramètres défini plus haut, en l'occurrence le vra jeu de paramètres
#On imagine que, sur un grand nombre de simus, au moins une doit conduire à un epsilon = 0 ?!?
#On constate qu'on obtient souvent des valeurs élevées...

#### Définition du prior comme distribution jointe a priori des paramètres ####

#Custom joint prior distribution for our parameters
@everywhere struct PriorDistribution <: ContinuousMultivariateDistribution end

@everywhere function Distributions.length(d::PriorDistribution)
    return 3 #Number of parameters
end


@everywhere function Distributions.rand(d::PriorDistribution)
    y = rand(Uniform(0,0.5)) #p0
    x = rand(Uniform(0,1)) #p2
    l= rand(Uniform(0,250.0))
    while x+y > 1 || y >= x
      y = rand(Uniform(0,0.5)) #p0
      x = rand(Uniform(0,1)) #p2
    end
    return [x,y,l] #Penser à avoir le même ordre que le modèle dummy
end

@everywhere function Distributions.rand!(d::PriorDistribution,vec_p::Array{Float64,1})
    vec_p[:] = rand(d)
end

@everywhere function Distributions.pdf(d::PriorDistribution, X::AbstractArray{T,1} where T=5) 
    (x,y,l) = X
    if 0.0 <= x <= 1.0
        if 0.0 <= y <= min(x,1.0-x)
            return 1.0*pdf(Uniform(0,250.0),l)
        else
            return 0.0
        end
    else
        return 0.0
    end
end

@everywhere function Distributions.insupport(d::PriorDistribution, x::Array{Float64,1}) 
    return pdf(d,x) != 0 # in support if non zero
end

# On génère un échantillon distribué suivant le prior
N_prior = 1_000
sample_prior = zeros(3, N_prior)
for i in 1:N_prior
    sample_prior[:, i] = rand(PriorDistribution())
end

parametric_model = ParametricModel(modelMPN, 
    [:p2, :p0, :lambda], #p2, p0, l'ordre correspond à celui dans lequel on a définit les params dans dummy
    PriorDistribution())


# On peut tester de calculer l'erreur dans le cas de paramètres choisis au hasard suivant le prior

set_param!(modelMPN, rand(PriorDistribution()));

@time epsilon = hybrid_dist_obs([simulate(modelMPN) for k in 1:100], vec_observations)
@show epsilon

mat_p = Matrix(CSV.read("/home/chirraneso/genealogy_abc_script/phylogeny/results_ET1_l_3/step_4/mat_p.csv", header=false,DataFrame))
vec_weights = Matrix(CSV.read("/home/chirraneso/genealogy_abc_script/phylogeny/results_ET1_l_3/step_4/weights.csv", header=false,DataFrame))[1:1000]
vec_dist = Matrix(CSV.read("/home/chirraneso/genealogy_abc_script/phylogeny/results_ET1_l_3/step_4/vec_dist.csv", header=false,DataFrame))[1:1000]
Random.seed!(7)
res_abc = abc_smc(parametric_model, vec_observations, 
                  hybrid_dist_obs, 
                nbr_particles = 1000, 
                tolerance = 50.0, #Il faudrait en choisir une plus faible
                #Il s'agit du critère d'arrêt (en termes de distance)
                alpha = 0.5,
                init_mat_p = mat_p, init_weights = vec_weights, init_vec_dist = vec_dist, #Ligne qui servira si on charge des précédents résultats
                save_iterations = true, 
                dir_results="/home/chirraneso/genealogy_abc_script/phylogeny/results_ET1_l_4/"
                )

histogram(res_abc.mat_p_end[1,:], weights=res_abc.weights, 
label="posterior",
title="p2",
alpha=0.8)

histogram!(sample_prior[1,:], weights= ones(N_prior)./N_prior, 
color=:grey,
label="prior", alpha=0.5)
savefig("//home/chirraneso/genealogy_abc_script/phylogeny/results_ET1_l_4/p2_plot.png")

histogram(res_abc.mat_p_end[2,:], weights=res_abc.weights, 
            label="posterior",
            title="p0",
            alpha=0.8
            )

histogram!(sample_prior[2,:], weights= ones(N_prior)./N_prior, 
    color=:grey,
    label="prior", alpha=0.5)
savefig("/home/chirraneso/genealogy_abc_script/phylogeny/results_ET1_l_4/p0_plot.png")

scatter(res_abc.mat_p_end[1,:], res_abc.mat_p_end[2,:], xlabel="p2", ylabel="p0", label="posterior",
        xlims=(-0.02,1), ylims=(-0.02,1), markersize=4)

scatter!(sample_prior[1,:], sample_prior[2,:], label="prior",
        color=:grey, marker=:cross, markersize=2)
savefig("/home/chirraneso/genealogy_abc_script/phylogeny/results_ET1_l_4/scatter_plot.png")

histogram(res_abc.mat_p_end[3,:], weights=res_abc.weights, 
            label="posterior",
            title="lambda",
            alpha=0.8
            )

histogram!(sample_prior[3,:], weights= ones(N_prior)./N_prior, 
    color=:grey,
    label="prior", alpha=0.5)
savefig("/home/chirraneso/genealogy_abc_script/phylogeny/results_ET1_l_4/lambda_plot.png")

#/home/chirraneso/genealogy_abc_script/phylogeny/