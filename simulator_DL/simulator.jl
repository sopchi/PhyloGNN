include("C:\\Users\\sophi\\OneDrive\\Documents\\3A_SDI\\projet_phylo\\code_stage\\phylogeny_simu_abc.jl")
include("C:\\Users\\sophi\\OneDrive\\Documents\\3A_SDI\\projet_phylo\\code_stage\\tree2adjmatrix.jl")
include("C:\\Users\\sophi\\OneDrive\\Documents\\3A_SDI\\projet_phylo\\code_stage\\simulator_scores.jl")
using Distributions
using Statistics
using PyCall
using Plots
using JSON

function class_q(q)
    return floor(10*q)
end

function class_delta(delta)
    return floor(10*delta)
end

function dico_simu(p0,p2, nb_leafs,t_max)
    q = p0/p2
    delta = p2 - p0
    delta_class = class_delta(delta)
    q_class =  class_q(q)

    dataset_dico = Dict()
    dataset_dico["p0"]=p0
    dataset_dico["p2"]=p2
    dataset_dico["delta"] = delta
    dataset_dico["q"] = q
    dataset_dico["q_class"]=q_class
    dataset_dico["delta_class"] = delta_class

    result=simulator_complete(lambda,p0,p2,nb_leafs,t_max)
        
    if result != "proba_0"
        #println(!isempty(findall(result[1].==nothing)))
        v_time=result[1]
        v_ancestor=result[2]
        count_iter=result[3]

        rscore = result[4]
        dataset_dico["rscore"] = rscore


        edge_list,root = to_edge_list(v_time,v_ancestor,t_max,lambda)
        edge_root = (root,"STM_00",0.00) # add a wild type stem cell for vizualisation
        phylogeny_simu= create_phylotree(edge_list,edge_root)

        dw1,paths1=tree2dyckword_step(phylogeny_simu.root.childs[1][1].childs[1][1],Dict([]),0)
        sort_dw1=triFusion_dw_step(phylogeny_simu.root.childs[1][1].childs[1][1],dw1,paths1,0)
        Y1=dyck_word2dyck_path(sort_dw1)

        dataset_dico["dyck"]  = Y1
        
        distances=get_absciss(phylogeny_simu.root,0,Dict())
        X1=sort(collect(distances))

        dataset_dico["LTT"] = X1

        adj = tree2adjmatrix(phylogeny_simu.root.childs[1][1],zeros(2*nb_leafs -1,2*nb_leafs -1),1)
        indexes = findall(!iszero, adj)
        row = Vector{Any}(nothing,size(indexes)[1])
        col = Vector{Any}(nothing,size(indexes)[1])
        
        for k in 1:size(indexes)[1]
            row[k] = indexes[k][1]
            col[k] = indexes[k][2]
            #node_features[k] =  adj[indexes[k][1],:]
        end
        data = adj[indexes]

        node_features = Vector{Any}(nothing,2*nb_leafs -1)
        jak2 = adj[1,1]
        adj[1,1]=0
        dist2root = Dict()
        dist2root[1] = jak2
        dist2parent = Dict()
        dist2parent[1] = jak2
        for node in 1:2*nb_leafs -1
            node_info = adj[node,:]
            if sum(node_info)!= 0.0
                index_childs = findall(x -> x!=0,node_info)

                desc1 = adj[node,index_childs[1]]
                dist2root[index_childs[1]] = dist2root[node] + desc1
                dist2parent[index_childs[1]] = desc1
                
                desc2 = adj[node,index_childs[2]]
                dist2root[index_childs[2]] = dist2root[node] + desc2
                dist2parent[index_childs[2]] = desc2
                
                node_features[node] = (dist2root[node],dist2parent[node],desc1,desc2)
            else
                node_features[node] = (dist2root[node],dist2parent[node],0,0)
            end
        end
        #adj_coo = sp.coo_matrix(adj)

        dataset_dico["coo"] = [data,row,col]
        dataset_dico["node_features"] = node_features
    else

        dataset_dico["rscore"] = 0
        dataset_dico["dyck"]  = 0
        dataset_dico["LTT"] = 0
        dataset_dico["coo"] = 0
        dataset_dico["node_features"] = 0

    end
    
    return dataset_dico 

end

""" node_features = Vector{Any}(nothing,2*nb_leafs -1)
for node in 1:2*nb_leafs -1
    node_info = adj[node,:]
    node_info = node_info[node_info.!=0]
    if node_info!= Float64[]
        node_features[node] = (node_info[1],node_info[2])
    else
        node_features[node] = (0,0)
    end
end"""