

function tree2adjmatrix(node,adjmatrix, index_parent)
    if isempty(node.childs[1])
        return
    end

    for child in node.childs
        index = findfirst(x -> x==0,sum(adjmatrix,dims=1))[2]
        adjmatrix[index_parent, index] = child[2]
        tree2adjmatrix(child[1],adjmatrix, index)
    end

    return adjmatrix
end

"""
n=5 #nb_leafs
ini =zeros(2*n +1,2*n +1)
ini[1,1]=1
M= tree2adjmatrix(phylogeny_simu.root.childs[1][1],ini,1)"""