using LinearAlgebra
using DataFrames
using CSV
using Random
using Combinatorics
using Distances 

include("tree_vfinale.jl")
include("STM_root.jl")
include("comparaison_metrics.jl")
include("extract_subtree.jl")
include("tree_reconstruction.jl")
include("dyck_metric.jl")
####### genealogie de reference ####

#p0=0.2
#p2=0.24

# leaves[61, 1, 35, 82, 78, 64, 30, 17, 77, 9]

t_max =115 +1
lambda=1
nb_leafs=10

topo_ref=Dict{Any, Any}("13.0" => 1, "2.0" => 4, "83.0" => 1, "3.0" => 1, "4.0" => 1, "1.0" => 10, "5.0" => 1)
ltt_ref=Dict{Any, Any}(78.0 => 1, 0.0 => 3, 47.0 => 1, 69.0 => 1, 98.0 => 1, 74.0 => 1, 115.0 => 10, 109.0 => 1, 71.0 => 1, 70.0 => 1, 15.0 => 1)
X_ref=Any[0.0, 15.0, 47.0, 69.0, 70.0, 71.0, 74.0, 78.0, 98.0, 109.0, 115.0]


function simulation(lambda,p0,p2,t_max,nb_leafs)
    """ simulation function"""
    count_iter=0 #time counter
    v_time= Vector{Any}(nothing,nb_leafs)
    v_ancestor= Vector{Any}(nothing,nb_leafs)
    v_particle=Vector{Any}(nothing,nb_leafs)

    #initialisation of first jak2 cell
    v_particle[1]=nb_leafs
    v_time[1]=1
    v_ancestor[1]=1

    while nothing in v_ancestor && count_iter<(t_max-lambda) #stops when nb of followed cell = nb leaves or when we reach the nulber max of iteration
        count_iter+=1

        #choose division according to parameter
        v_division=rand!(zeros(nb_leafs))
        v_division[v_division.<=p2/(1-p0)].=2
        v_division[p2/(1-p0) .<v_division.<=1].=1

        index_2=findall(v_division.==2)
        #index_1=findall( v_division.==1)

        div_sym=index_2[findall(x->(x != nothing) && (x>=2),v_particle[index_2])]

        #extinction=index_0[findall(x->(x != nothing) && (x>=2),v_particle[index_0])]
        #extinction=index_0[findall(v_particle[index_0].!=nothing)]
        for cell in div_sym
            if rand() > p0/p2 #no extinction
                x=rand(0:(v_particle[cell]))
                if x !=0 && x!= v_particle[cell]
                    v_particle[cell]-=x
                    # create new cell
                    new_cell=findfirst(v_particle.==nothing)
                    v_particle[new_cell]=x
                    v_ancestor[new_cell]=cell
                    v_time[new_cell]=count_iter
                end
                if !(nothing in v_ancestor)
                    return v_time, v_ancestor,v_particle,count_iter # needed ? choisi le premier , biais ? 
                end
            end
        end
    end
            return v_time, v_ancestor,count_iter
        
end

function simu_complete(lambda,p0,p2,nb_leafs,t_max)
    """ function that launch the simulation function as many times as needed"""
        iter=0
        result = simulation(lambda,p0,p2,t_max,nb_leafs)
        while !isempty(findall(result[1].==nothing)) && iter < 100
            result = simulation(lambda,p0,p2,t_max,nb_leafs)
            iter+=1
        end
        if iter >=100
            return "proba_0"
        end
        return result
    end


function compute_distance(lambda,p0,p2,nb_leafs,t_max,Y_ref) #topo_ref,ltt_ref,X_ref)
    result=simu_complete(lambda,p0,p2,nb_leafs,t_max)
    if result != "proba_0"

    #println(!isempty(findall(result[1].==nothing)))
        v_time=result[1]
        v_ancestor=result[2]
        count_iter=result[3]

        edge_list,root = to_edge_list(v_time,v_ancestor,t_max,lambda)
        edge_root = (root,"STM_00",0.00) # add a wild type stem cell for vizualisation
        phylogeny_simu= create_phylotree(edge_list,edge_root)

        dw1,paths1=tree2dyckword_step(phylogeny_simu.root.childs[1][1].childs[1][1],Dict([]),0)
        sort_dw1=triFusion_dw_step(phylogeny_simu.root.childs[1][1].childs[1][1],dw1,paths1,0)
        Y1=dyck_word2dyck_path(sort_dw1)

        return minkowski(Y1,Y_ref,1) #colijn*10 + D_ltt
    else 
        return Inf
    end
    """
    topo_simu=label_recurs(phylogeny_simu.root.childs[1][1].childs[1][1],Dict([]))[2]
    colijn = colijn_metrics(topo_simu,topo_ref)

    ltt_simu=get_absciss(phylogeny_simu.root,0,Dict([])) # avec SET
    X_simu=sort(collect(keys(ltt_simu))) # add keys() for dict
    D_ltt=LTT_distance(X_simu,X_ref,ltt_simu,ltt_ref)"""

    
end


function compute_mean_distance(lambda,p0,p2,nb_leafs,t_max,Y_ref,repet) #topo_ref,ltt_ref,X_ref)
    mean = 0
    for k in 1:repet
        mean += compute_distance(lambda,p0,p2,nb_leafs,t_max,Y_ref)
    end
    return mean/repet
"""
    topo_simu=label_recurs(phylogeny_simu.root.childs[1][1].childs[1][1],Dict([]))[2]
    colijn = colijn_metrics(topo_simu,topo_ref)

    ltt_simu=get_absciss(phylogeny_simu.root,0,Dict([])) # avec SET
    X_simu=sort(collect(keys(ltt_simu))) # add keys() for dict
    D_ltt=LTT_distance(X_simu,X_ref,ltt_simu,ltt_ref)"""

    #return minkowski(Y1,Y_ref,1) #colijn*10 + D_ltt
end
### commande a lancer 
#compute_distance(lambda,p0,p2,nb_leafs,t_max,topo_ref,ltt_ref,X_ref)# en choisissant p0 et p2 




