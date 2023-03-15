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
#include("c:\\Users\\sophi\\phylogenetic_tree\\code_arbre\\stage\\stage_sophia\\tree_reconstruction.jl")"""

function get_leaves(tree)
    """ get all leaves name"""
    leaves=Any[]
    for node in tree.nodes
        if !occursin("Inner",node.name) && !occursin("STM_00",node.name) && !occursin("root",node.name) && !occursin("jak2",node.name)
            push!(leaves,node.name)
        end
    end
    return leaves
    
end


function recurs_paths(node,path,paths,sub_nodes)
    """ find path of sub tree (tree with sub nodes = leaves) """
    if node.name in sub_nodes # leaf in sub tree
        for edge in path
            push!(paths,edge)
        end
        return paths
    elseif isempty(node.childs[1]) # leaf not in sub tree
        return
    end

    for child in node.childs
        new_path=copy(path)
        push!(new_path,(node.name,child[1].name,child[2]))
        recurs_paths(child[1],new_path,paths,sub_nodes)
    end
    return paths
end

function delete_innernode(node,parent)
 """ function to delete inner node with 1 child after extraction """
    if isempty(node.childs[1]) #leaf
        return 
    
    elseif size(node.childs)[1]==1 # inner node to delete
        if parent.childs[1][1].name == node.name # inner node to delete is on the left, index =1
            if size(parent.childs)[1]==1
                change= [(node.childs[1][1],node.childs[1][2]+parent.childs[1][2])]
                parent.childs=change
            else
                change= [(node.childs[1][1],node.childs[1][2]+parent.childs[1][2]),(parent.childs[2][1],parent.childs[2][2])]
                parent.childs=change
            end
        else # inner node to delete is on the right, index =2
            change= [(parent.childs[1][1],parent.childs[1][2]),(node.childs[1][1],node.childs[1][2]+parent.childs[2][2])]
            parent.childs=change
        end
        delete_innernode(node.childs[1][1],parent)

    else
        delete_innernode(node.childs[1][1],node)
        delete_innernode(node.childs[2][1],node)
    end
end
"""
#tree_VE= full tree
leaves=get_leaves(tree_VE)
all_combi=collect(combinations(leaves, 5))
jak=tree_VE.root.childs[1][1]
edges= recurs_paths(jak,Set([]),Set([]),all_combi[1])
edge_root = ("jak2","STM_00",0.00) # add a wild type stem cell for vizualisation
tree_VE_sub= create_phylotree(edges,edge_root)

delete_innernode(tree_VE_sub.root.childs[1][1].childs[1][1],tree_VE_sub.root.childs[1][1]) #node = first inner , parent = jak2

tree_viz = to_clade(tree_VE_sub.root,0.00)
bp.draw(tree_viz)"""