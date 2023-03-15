using LinearAlgebra
using DataFrames
using CSV
using Random
using Combinatorics
using Distances 
"""
include("tree_vfinale.jl")
include("STM_root.jl")
include("comparaison_metrics.jl")
include("extract_subtree.jl")
include("tree_reconstruction.jl")
include("dyck_metric.jl")"""
####### genealogie de reference ####

"""
p0=0.2
p2=0.22

t_max =200
lambda=1
nb_leafs=4
"""

function simulator(lambda,p0,p2,t_max,nb_leafs)
    """ simulation function"""
    count_iter=0 #time counter
    v_time= Vector{Any}(nothing,nb_leafs)
    v_ancestor= Vector{Any}(nothing,nb_leafs)
    v_particle=Vector{Any}(nothing,nb_leafs)
    r_score = Vector{Any}(nothing,t_max-lambda)
    score_value=1

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
        
        proba = v_division[findall(x->(x != nothing),v_particle)]
        proba[proba .==2] .= p2/(1-p0)
        proba[proba.==1] .= 1- (p2/(1-p0))
       
        score_value*=prod.(eachcol(proba))[1]

        div_sym=index_2[findall(x->(x != nothing) && (x>=2),v_particle[index_2])]


        score_value*= 1 #r_score particule 


        for cell in div_sym
            if rand() > p0/p2 #no extinction
                score_value*= 1-(p0/p2)
                x=rand(0:(v_particle[cell]))
                score_value*=1/(v_particle[cell] +1)
                if x !=0 && x!= v_particle[cell]
                    v_particle[cell]-=x
                    # create new cell
                    new_cell=findfirst(v_particle.==nothing)
                    v_particle[new_cell]=x
                    v_ancestor[new_cell]=cell
                    v_time[new_cell]=count_iter
                end
                if !(nothing in v_ancestor)
                    return v_time, v_ancestor,v_particle,count_iter, r_score[r_score.!=nothing] # needed ? choisi le premier , biais ? 
                end
            else
                score_value*= p0/p2
            end
        end
        #println(score_value)
        r_score[count_iter]=score_value
        #println(r_score)
    end

    return v_time, v_ancestor, count_iter, r_score
        
end

function simulator_complete(lambda,p0,p2,nb_leafs,t_max)
    """ function that launch the simulation function as many times as needed"""
        iter=0
        result = simulator(lambda,p0,p2,t_max,nb_leafs)
        while !isempty(findall(result[1].==nothing)) && iter < 100
            result = simulator(lambda,p0,p2,t_max,nb_leafs)
            iter+=1
        end
        if iter >=100
            return "proba_0"
        end
        return result
end

