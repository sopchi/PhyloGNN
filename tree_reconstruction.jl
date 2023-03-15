"""using LinearAlgebra
using PyCall
using DataFrames
using CSV
using Intervals 

bp=pyimport("Bio.Phylo")
st=pyimport("io")
mpl = pyimport("matplotlib")
plt = pyimport("matplotlib.pyplot")
nx=pyimport("networkx")
bt=bp.BaseTree

include("simu_finale.jl")
include("tree_vfinale.jl")"""

function to_edge_list(v_time,v_ancestor,age_sampling,lambda)
    jak2=1 # default, at the beginning no asymetric division 
    nb_leafs= size(v_time)[1]
    name=Any[string(k) for k in 1:nb_leafs]
    edge_list=Set([])
    iter = 1

    time=findmax(v_time)[1] # duration of simulation
    inner_join=[ time for k in 1:nb_leafs]
    time_blood=age_sampling-lambda- findmax(v_time)[1] # length to add at the end of leaves branches 
    
    while (nb_leafs -size(findall(v_time.==0))[1]) != 2 # exit when it only stays the first jak2 cell and its descendant
        
        youngest=findmax(v_time)[2] # find the youngest cell
        ancetre=v_ancestor[youngest] # find its ancestor
        same_ancetre=findall(v_ancestor .== ancetre) # find all cell with the same ancestor

        if youngest in same_ancetre
            deleteat!(same_ancetre,findall(x->x==youngest,same_ancetre)) #exclude the cell to be added to the tree
        end

        if occursin("Inner",name[youngest]) # the cell is an inner node
            push!(edge_list,(name[youngest],"Inner"*string(iter),1+ inner_join[youngest] - v_time[youngest])) 
        else # the cell is a leaf
            push!(edge_list,(name[youngest],"Inner"*string(iter),1 + time - v_time[youngest]+ time_blood))
        end 
        
        if occursin("Inner",name[ancetre]) # the cell's ancestor is an inner node
            push!(edge_list,(name[ancetre],"Inner"*string(iter),1+ inner_join[ancetre] - v_time[youngest] )) #create an edge
        else # the cell's ancestor is a leaf
            push!(edge_list,(name[ancetre],"Inner"*string(iter),1 + time - v_time[youngest] + time_blood)) #create an edge
        end 

        inner_join[ancetre]=v_time[youngest]-1 #update ancestor info to compute branch length

        #delete all info about the cell that was added
        v_time[youngest]=0 
        v_ancestor[youngest]=0
        name[youngest]=0

        #ancestor is now an inner node
        name[ancetre]="Inner"*string(iter)
        
        iter+=1

    end
    last=findall(v_time.!=0)
    # identify first jak2 cell and its descendant
    if v_ancestor[last[1]]==last[1]
        youngest=last[2]
        ancetre=last[1]
    else
        youngest=last[1]
        ancetre=last[2]
    end

    # add edges
    if occursin("Inner",name[youngest])
        push!(edge_list,(name[youngest],"Inner"*string(iter),1+ inner_join[youngest] - v_time[youngest]))
    else
        push!(edge_list,(name[youngest],"Inner"*string(iter),1 + time - v_time[youngest] + time_blood))
    end 
    
    if occursin("Inner",name[ancetre])
        push!(edge_list,(name[ancetre],"Inner"*string(iter),1+ inner_join[ancetre] - v_time[youngest]))
    else
        push!(edge_list,(name[ancetre],"Inner"*string(iter),1 + time - v_time[youngest]+ time_blood))
    end 
    

    jak2=v_time[youngest] -1 #give the number of asymetric division between the first jak2 cell and its first descendant at the final iteration

    name[ancetre]="Inner"*string(iter)
    
    push!(edge_list,("jak2",name[ancetre],jak2 +lambda))  #edge between the first jak2 cell and its first descendant + betwenn the forst jak2 cell and birth
    return edge_list, "jak2"

end

"""
########## EXAMPLE 
############ PIPELINE #############

lambda = 10 #apparition of jak2, in month
p0 = 0.6
p2 = 0.3
p1 = 1- p2 - p0
t_max = 40 #age sampling, in month
nb_leafs = 12

println("start")

result = simu_complete(lambda,p0,p2,t_max,nb_leafs)

v_time=result[1]
v_ancestor=result[2]
v_particle=result[3]
count_iter=result[4]

println(v_time)
println(v_ancestor)
println(v_particle)
println(count_iter)

### simulation outputs 
#v_time=Any[5, 1, 4, 5, 3, 5, 6, 6, 6, 6]
#v_ancestor=Any[2, 2, 2, 3, 2, 5, 1, 3, 4, 6]
#v_particle=Any[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

age_sampling=40 #age at blood sampling, in month
lambda=10 # age of apparition of jak2 mutation, in month

edge_list,root = to_edge_list(v_time,v_ancestor,age_sampling,lambda)
edge_root = (root,"STM_00",0.00) # add a wild type stem cell for vizualisation
tree_VE= create_phylotree(edge_list,edge_root)

tree_viz = to_clade(tree_VE.root,0.00)
bp.draw(tree_viz)

"""