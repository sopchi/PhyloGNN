"""bp=pyimport("Bio.Phylo")
st=pyimport("io")
mpl = pyimport("matplotlib")
plt = pyimport("matplotlib.pyplot")
nx=pyimport("networkx")
bt=bp.BaseTree

using LinearAlgebra
using PyCall
using DataFrames
using CSV
using Intervals """



function hamming(l1,l2)
    """ compute the hamming distance between two genomes"""
    A=(l1-l2).^2
    return size(A[A.==1])[1]
end 

function dist_matrix(genet_matrix,n)
    """ compute the distance matrix of a genome matrix"""
    matrix=Matrix{Float64}(undef,n,n)
    for i in 1:n, j in i:n
            h=hamming(genet_matrix[i,:],genet_matrix[j,:])
                matrix[i,j]=h 
                matrix[j,i]=h # the distance matrix is symmetric
    end
    return matrix
end 

function find_min_pair(M,S)
    """ find the two closest cells according to the NJ formula
    M is the distance matrix and S is the vector that computes 
    for each cell the sum of its distances divided by the number 
    of remaining cells to add to the tree minus 2 """

    n= size(M)[1]
    N=copy(M)
    for i in 1:n, j in 1:n
        N[i,j]-=S[i]+S[j] # NJ formula 
    end
    x = findmin(N)[2] #tuple of indices of the pair of cell to join
    d = M[x[1],x[2]] # the distance of the pair of cell to be joined, needed to compute the branch length
    return d , x
end 

function nj_tree(M, name)
    """ construct the list of edges of the unrooted phylogenetic tree
    an edge is stored as a tuple containing the two cells' names and the length of the edge """

    T=Set([]) #initialisation of the edges list 
    name_copie=copy(name)
    k=0

    while size(M)[1]-2 !=0 # stops when there are only two cells remaining
        n=size(name_copie)[1] # number of remaining cells and inner node
        S = sum(M, dims = 2)./(n-2) # the S of the NJ formula
    
        M[M.==0].= Inf # set the diagonal to Inf 
    
        d, x = find_min_pair(M,S) # get the indices of the pair of cell to join and their distance
        i,j=x[1],x[2]
        
        # create the inner node that joins the two cells, and the two corresponding edges to the list of edges
        push!(T, ("Inner"*string(k),name_copie[i],(d+ S[i]- S[j])/2))
        push!(T, ("Inner"*string(k),name_copie[j],(d+ S[j]- S[i])/2)) 
        
        # create the inner node in the distance matrix and its distances to the other cells of the matrix 
        temp= M[i,:] + M[j,:]
        temp.-=d
        temp./=2
    
        M[j,:]=temp
        M[:,j]=temp
        M=M[Not([i]),Not([i])] # drop one of the cell that was add to the tree
    
        M[diagind(M)].=0 # set the diagonal to 0
    
        deleteat!(name_copie,i) # update the name of the cell to join
        name_copie[j]="Inner"*string(k)
        k+=1
    end 
    
        push!(T,(name_copie[1],name_copie[2],M[1,2])) # join the last two cells by a edge of length the distance of the elements in the distance matrix 
        
    return T
end


function find_root(T)
    """ find the longest edge to insert the root in it """
    max_d=0
    root_edge=""
    for edge in T
            if edge[3] > max_d
                max_d= edge[3]
                root_edge=edge
            end
    end 
    delete!(T,root_edge) # delete the longest edge of the list of edges 
    return root_edge,T
end

"""Type Node with two attributes, a name and a list of tuples containing each a child of type Node with the length of the branch """
mutable struct Node
    name::String 
    childs::Vector{Tuple}
end

"""Type Phylotree with two attributes, the root of the tree of type Node, a set of all the nodes of the tree"""
mutable struct PhyloTree
    root::Node 
    nodes::Set
end

function find_child(name,T,edge_root,add_name)
    """ find the child of a node according to its name """

    childs=[]
    if name =="root"
        # root's childs are the cell separated by the edge_root, insert the root inside the edge at equidistance from childs
        return [(edge_root[1],edge_root[3]/2), (edge_root[2],edge_root[3]/2)]
    end

    for edge in T
        if edge[1] == name || edge[2] == name # find edges where node name is one of the extremity
            if (edge[1] != name) && !(edge[1] in add_name)
                push!(childs,(edge[1],edge[3])) # identify the other extremity as a child of node name if it has not been added yet to the tree 
            elseif (edge[2] != name) && !(edge[2] in add_name)
                push!(childs,(edge[2],edge[3]))
            end
        end
    end
    return childs
end

function create_node(name,T,edge_root,nodes,add_name)
    """ recursiv function to create a node of type Node"""

    childs=find_child(name,T,edge_root,add_name) # find childs of the node name 

    if isempty(childs) # if it has no childs, it's a leaf 
        node = Node(name, [()])
    else 
        for child in childs
            push!(add_name,child[1])
        end
        l=[]
        for child in childs
            c=create_node(child[1],T,edge_root,nodes,add_name) # create childs node 
            push!(l,(c,child[2])) # child[2] id the branch length 
        end
        node = Node(name, l) # create the node name, where l is the list of tuples containing each a child of type Node with the length of the branch
    end
        
    push!(nodes,node) # update the nodes list of all nodes in the tree 
    return node
end

function create_phylotree(T, edge_root)
    """ create a phylogenetic tree"""
    nodes=Set([])
    add_name=["root"]
    root= create_node("root",T,edge_root,nodes,add_name)
    tree = PhyloTree(root,nodes)
    return tree
end

function to_clade(node,l)
    """ transform a phylotree into a tree of clades for vizualisation with Biopython,
    start the recursiv function with the root of the phylotree ,
    l is the branch length linking the node to the rest of the tree """

    childs=node.childs
    if isempty(childs[1])
        return bt.Clade(l)
    else
        clade_l=[]
        for child in childs
            push!(clade_l,to_clade(child[1],child[2]))
        end
        return bt.Clade(l, clades=clade_l)
    end
end

function recurs_root(node,d,distances)
    """ find all different distances separting leaves to root """
    if node.name == "root"
        change= [(node.childs[1][1],0),(node.childs[2][1],0)] # reset the root position
        node.childs=change
    end

    if isempty(node.childs[1]) # True if the noe is a leaf
        push!(distances,d)
        return distances
    end

    # if the node is not a leaf
    # save branch lengths
    d1 = node.childs[1][2] 
    d2 = node.childs[2][2]
    # go down the tree
    recurs_root(node.childs[1][1],d+d1,distances)
    recurs_root(node.childs[2][1],d+d2,distances)
    
end

function root_length(T)
    """compute the position of the root inside a given edge such as it is approximately at equidistance from all leaves"""
    dist= T.root.childs[1][2]*2 # branch length of the edge root 
    distances=recurs_root(T.root,0,Set([])) # get all the distances separating leaves of root

    # create intervals of all the possible distance to root 
    intervals=[]
    for d in distances
        push!(intervals,Interval{Open, Open}(d, d+dist))
    end

    # find the smallest interval,often the root is to put inside an edge which one of its end is a leaf
    min_i=intervals[1]
    for i in intervals
        if i < min_i
            min_i=i
        end
    end

    # compute the intersection of all intervals
    inter=min_i
    for i in intervals
        if intersect(i,inter) != Interval{Float64, Open, Open}(0.0, 0.0) # if the intersection is empty (ludicrous values) do not take it into account
            inter = intersect(i,inter)
        end
    end

    d_root = inter.first # get the position of the root

    change= [(T.root.childs[1][1],dist-d_root),(T.root.childs[2][1],d_root)] # put the root at its right place
    T.root.childs=change

    return T
end 

"""
############ VAN EGEREN TREE #############

snp_df = CSV.read("C:\\Users\\sophi\\phylogenetic_tree\\snv_patient1.csv",normalizenames=true,DataFrame)
genet_df = select!(snp_df, Not(:Chromosome))
genet_df = select!(snp_df, Not(:Position))
genet_df = select!(snp_df, Not(:counts))
genet_matrix = transpose(Matrix{Float64}(genet_df))
n=size(genet_matrix)[1]
dist_VE=dist_matrix(genet_matrix,n)
cell_name = names(genet_df)
T_VE=nj_tree(dist_VE,cell_name)
edge_root_VE,Tree_VE=find_root(T_VE)
tree_VE= create_phylotree(Tree_VE,edge_root_VE)
rooted_VE=root_length(tree_VE)
tree_viz = to_clade(rooted_VE.root,1)
bp.draw(tree_viz)


################ WILLIAMS TREE ############

snp_df = CSV.read("C:\\Users\\sophi\\phylogenetic_tree\\code_arbre\\stage\\stage_sophia\\snv_PD9478.csv",normalizenames=true,DataFrame)
genet_df = select!(snp_df, Not(:Column1))
genet_matrix = transpose(Matrix{Float64}(genet_df))
n=size(genet_matrix)[1]
dist_VE=dist_matrix(genet_matrix,n)
cell_name = names(genet_df)
T_VE=nj_tree(dist_VE,cell_name)
edge_root_VE,Tree_VE=find_root(T_VE)
tree_VE= create_phylotree(Tree_VE,edge_root_VE)
rooted_VE=root_length(tree_VE)
tree_viz = to_clade(rooted_VE.root,1)
bp.draw(tree_viz)
"""