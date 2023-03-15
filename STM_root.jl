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

include("tree_vfinale.jl")"""

function add_origin(edge_list, origin_name)
    """ function to add the origin edge"""

    for edge in edge_list
        if edge[1]==origin_name || edge[2]==origin_name # origin name in dataset 
            delete!(edge_list,edge)
            if edge[1]==origin_name
                push!(edge_list,(edge[2],"jak2",edge[3]))
            else
                push!(edge_list,(edge[1],"jak2",edge[3]))
            end
        end
    end

    edge_root = ("jak2",origin_name,0.00)
    return edge_root
end

"""
jak2_df=CSV.read("C:\\Users\\sophi\\phylogenetic_tree\\snv_patient1.csv",normalizenames=true,DataFrame)
jak2_genet = select!(jak2_df, Not(:Chromosome))
jak2_genet = select!(jak2_df, Not(:Position))
jak2_genet = select!(jak2_df, Not(:counts))
cell=names(jak2_df)

jak2_cell= [x for x in jak2_df[12259,:]]
jak2_cell=jak2_cell.*[k for k in 1:42]

jak2_df=jak2_df[:,jak2_cell[jak2_cell.!=0]] 
jak2_df=hcat(jak2_df,jak2_genet[:,"STM_00"])
println(names(jak2_df))

jak2_matrix = transpose(Matrix{Float64}(jak2_df))
n=size(jak2_matrix)[1]
dist_jak2=dist_matrix(jak2_matrix,n)

cell_jak2 = names(jak2_df)
T_jak2=nj_tree(dist_jak2,cell_jak2)

edge_root= add_origin(T_jak2, "x1")
tree_jak2= create_phylotree(T_jak2,edge_root)

tree_viz = to_clade(tree_jak2.root,0)
bp.draw(tree_viz)"""

"""
########### WILLIAMS
snp_df = CSV.read("C:\\Users\\sophi\\phylogenetic_tree\\code_arbre\\stage\\stage_sophia\\snv_PD6646.csv",normalizenames=true,DataFrame)
genet_df = select!(snp_df, Not(:Column1))
stm_df=DataFrame("STM_00" => [0 for i in 1:size(genet_df)[1]])
genet_df = hcat(genet_df,stm_df)
genet_matrix = transpose(Matrix{Float64}(genet_df))
n=size(genet_matrix)[1]
dist_VE=dist_matrix(genet_matrix,n)
cell_name = names(genet_df)
T_VE=nj_tree(dist_VE,cell_name)
edge_origin=""
for edge in T_VE
    if edge[1]=="STM_00" || edge[2]=="STM_00"
        edge_origin=edge
        delete!(T_VE,edge)
    end
end

tree_VE= create_phylotree(T_VE,edge_origin)
tree_viz = to_clade(tree_VE.root,1)
bp.draw(tree_viz)

############ VAN EGEREN #####
snp_df = CSV.read("C:\\Users\\sophi\\phylogenetic_tree\\snv_patient1.csv",normalizenames=true,DataFrame)
genet_df = select!(snp_df, Not(:Chromosome))
genet_df = select!(snp_df, Not(:Position))
genet_df = select!(snp_df, Not(:counts))
genet_matrix = transpose(Matrix{Float64}(genet_df))
n=size(genet_matrix)[1]
dist_VE=dist_matrix(genet_matrix,n)
cell_name = names(genet_df)
T_VE=nj_tree(dist_VE,cell_name)
edge_origin=""
for edge in T_VE
    if edge[1]=="STM_00" || edge[2]=="STM_00"
        edge_origin=edge
        delete!(T_VE,edge)
    end
end

tree_VE= create_phylotree(T_VE,edge_origin)

tree_viz = to_clade(tree_VE.root,1)
bp.draw(tree_viz)"""