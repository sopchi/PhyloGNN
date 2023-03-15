"""using LinearAlgebra
using PyCall
using DataFrames
using CSV
using Intervals 
using Combinatorics

bp=pyimport("Bio.Phylo")
st=pyimport("io")
mpl = pyimport("matplotlib")
plt = pyimport("matplotlib.pyplot")
nx=pyimport("networkx")
bt=bp.BaseTree

include("c:\\Users\\sophi\\phylogenetic_tree\\code_arbre\\stage\\stage_sophia\\tree_vfinale.jl")
include("c:\\Users\\sophi\\phylogenetic_tree\\code_arbre\\stage\\stage_sophia\\tree_reconstruction.jl")
include("c:\\Users\\sophi\\phylogenetic_tree\\code_arbre\\stage\\stage_sophia\\STM_root.jl")"""

function label_recurs(node,dico)
    """ function that relabel tree for colijn metric computation"""
    if isempty(node.childs[1]) # leaf
        node.name="1.0"
        if haskey(dico,node.name)
            dico[node.name]+=1
        else
            push!(dico,node.name=>1)
        end

    else
        s1=parse(Float64 ,label_recurs(node.childs[1][1],dico)[1].name) # get child node  new label
        s2=parse(Float64 ,label_recurs(node.childs[2][1],dico)[1].name) # get child node new label 

        node.name=string(1/2*(max(s1,s2))*(max(s1,s2)-1)+ min(s1,s2) +1) #colijn formula 
        if haskey(dico,node.name)
            dico[node.name]+=1
        else
            push!(dico,node.name=>1)
        end
    end
    return node, dico
    
end

function colijn_metrics(labels_1,labels_2)
    """ compute colijn metric
    labels_1 is a dict where keys are label and values number of element in tree with label"""
    distance=0

    in_both= intersect(Set(keys(labels_1)),Set(keys(labels_2))) # common label
    for label in in_both
        distance += abs(labels_1[label] - labels_2[label])
    end
    
    key_1=setdiff(Set(keys(labels_1)),in_both) # label only in tree 1 
    key_2=setdiff(Set(keys(labels_2)),in_both) # label only in tree 2 

    for i in key_1
        distance +=labels_1[i]
    end
    for j in key_2
        distance +=labels_2[j]
    end

    return distance
end

function get_absciss(node,d,distances)
    """ function that return dict of all different branch lenghts and nb of repetition of each"""
    if node.name == "root"
        #push!(distances,0)
        if haskey(distances,0)
            distances[0]+=1
        else
            push!(distances,0=>1)
        end
    end

    if isempty(node.childs[1]) # leaf
        #push!(distances,d)
        
        """if haskey(distances,d)
            distances[d]+=1
        else
            push!(distances,d=>1)
        end"""
        return distances
    end

    for child in node.childs
        
        #push!(distances,d+child[2])
        
        if haskey(distances,d+child[2])
            distances[d+child[2]]+=1
        else
            push!(distances,d+child[2]=>1)
        end
        get_absciss(child[1],d+child[2],distances)
    end
    return distances
end

function LTT_distance(X1,X2,distances1,distances2)
    """ compute LTT distance : area betwenn 2 LTT plots"""
    grid = sort(collect(union(Set(X1),Set(X2)))) # absciss scale 
    height1=0
    height2=0
    area=0
    #### prolonge shortest tree 
    if last(grid) in X2
        push!(X1,last(grid))
        push!(distances1,last(grid)=>0)
    else
        push!(X2,last(grid))
        push!(distances2,last(grid)=>0)
    end
    #####
    
    for i in 2:size(grid)[1]-1 # - 2 if stop before last
        width=grid[i+1]-grid[i]
        if grid[i] in X1
            height1+= distances1[grid[i]] # 1 if set 
        end
    
        if grid[i] in X2
            height2+= distances2[grid[i]] # 1 if set 
        end
        area+=abs(width*height1 - width*height2)
    end

    return area
    
end

function find_min_distance(n,leaves,tree_simu,d_exp,distances_exp)
    all_combi=combinations(leaves, n)
    X2=sort(collect(keys(distances_exp))) # add keys() for dict
    jak=tree_simu.root.childs[1][1]
    final_simu_tree=nothing
    min_dist=Inf
    min_colijn=Inf
    min_LTT=Inf
    min_X1=[]
    min_distances_simu=[]

    for combi in all_combi
        edges= recurs_paths(jak,Set([]),Set([]),combi)
        edge_root = ("jak2","STM_00",0.00) # add a wild type stem cell for vizualisation
        tree_simu_sub= create_phylotree(edges,edge_root)
        delete_innernode(tree_simu_sub.root.childs[1][1].childs[1][1],tree_simu_sub.root.childs[1][1]) #node = first inner , parent = jak2
    
        d_simu=label_recurs(tree_simu_sub.root.childs[1][1].childs[1][1],Dict([]))[2]
        colijn = colijn_metrics(d_simu,d_exp)
    
        distances_simu=get_absciss(tree_simu_sub.root,0,Dict([])) # avec SET
        X1=sort(collect(keys(distances_simu))) # add keys() for dict
        #X1.*=12 # ajustement 
        D_ltt=LTT_distance(X1,X2,distances_simu,distances_exp) #LTT_distance(X1,X2) if set
    
        if D_ltt + colijn < min_dist
            min_dist=D_ltt + colijn
            min_colijn = colijn
            min_LTT=D_ltt
            min_X1=X1
            final_simu_tree = create_phylotree(edges,edge_root)
            min_distances_simu=distances_simu

        end
    end
    return min_dist,min_colijn,min_LTT,final_simu_tree,min_X1,X2,min_distances_simu
end
"""
######## PIPELINE

v_time=Any[3, 1, 9, 6, 7, 7, 10]
v_ancestor=Any[2, 2, 1, 2, 1, 4, 5]

age_sampling=40 #age at blood sampling, in month
lambda=8 # age of apparition of jak2 mutation, in month

edge_list,root = to_edge_list(v_time,v_ancestor,age_sampling,lambda)
edge_root = (root,"STM_00",0.00) # add a wild type stem cell for vizualisation
tree_VE= create_phylotree(edge_list,edge_root)

dico=label_recurs(tree_VE.root.childs[1][1].childs[1][1],Dict([]))[2]

D=colijn_metrics(dico1,dico2)

# pas de notion de topologie 
distances=get_absciss(tree1.root,0,Set([]))
X1=sort(collect(distances))


distances2=get_absciss(tree2.root,0,Set([]))
X2=sort(collect(distances2))
Y1=[k for k in 0:size(X1)[1]-1]
Y2=[k for k in 0:size(X2)[1]-1]

plt.step(X1,Y1)
plt.step(X2,Y2)
plt.show()

D_ltt=LTT_distance(X1,X2)"""