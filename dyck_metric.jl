###### only topo ########
function tree2dyckword(node,paths)
  """ dyck word where 1 = up and 0 = down"""
    if isempty(node.childs[1]) # leaf
        push!(paths,node.name=>1)
        return [1] #[node.name]
    else
        L=vcat(1,tree2dyckword(node.childs[2][1],paths)[1]) # left first
        #vcat([node.name],in_order(node.childs[2][1]))
        R=tree2dyckword(node.childs[1][1],paths)[1] # right then 
        push!(paths,node.name=>size(vcat(vcat(L,[node.name]),vcat(R,[node.name])))[1])
    end

    return vcat(vcat(L,0),vcat(R,0)),paths 
    #vcat(vcat(L,[node.name]),vcat(R,[node.name]))
end

function triFusion_dw(node,dyck_word,paths)
  """ tri fusion algorithm inspirated """
    if size(dyck_word)[1] <= 1
      return dyck_word
    else
      n=size(dyck_word)[1]
      child_l=node.childs[2][1] # of type Node
      child_r=node.childs[1][1] # of type Node
      # fusion ( sub_dyckword of left part , sub_dyckword of right part)
      return fusion_dw(triFusion_dw(child_l,dyck_word[2:paths[child_l.name]+1],paths), triFusion_dw(child_r,dyck_word[paths[child_l.name]+ 3:n-1],paths))
    end
end

function fusion_dw(dw1, dw2)
    if size(dw1)[1] < size(dw2)[1] # need to invert because right has more embranchements 
      return vcat(vcat(1,dw2),vcat(vcat(0,dw1),0))
    else
      return vcat(vcat(1,dw1),vcat(vcat(0,dw2),0))
    end
end


########## topo + branch length #######
function tree2dyckword_step(node,paths,high)
  """ dyck word where high = up and 0 = down"""
  if isempty(node.childs[1]) #leaf
      push!(paths,node.name=>1)
      return [high] #[node.name]
  else
      L=vcat(high,tree2dyckword_step(node.childs[2][1],paths,node.childs[2][2])[1])
      #vcat([node.name],in_order(node.childs[2][1]))
      R=tree2dyckword_step(node.childs[1][1],paths,node.childs[1][2])[1]
      push!(paths,node.name=>size(vcat(vcat(L,[node.name]),vcat(R,[node.name])))[1])
  end

  return vcat(vcat(L,0),vcat(R,0)),paths
  #vcat(vcat(L,[node.name]),vcat(R,[node.name]))
end

function triFusion_dw_step(node,dyck_word,paths,high)
  if size(dyck_word)[1] <= 1
    return dyck_word
  else
    n=size(dyck_word)[1]
    child_l=node.childs[2][1]
    child_r=node.childs[1][1]
    return fusion_dw_step(triFusion_dw_step(child_l,dyck_word[2:paths[child_l.name]+1],paths,node.childs[2][2]), triFusion_dw_step(child_r,dyck_word[paths[child_l.name]+ 3:n-1],paths,node.childs[1][2]),high)
  end
end

function fusion_dw_step(dw1, dw2,ancestor)
  if size(dw1)[1] < size(dw2)[1]
    return vcat(vcat(ancestor,dw2),vcat(vcat(0,dw1),0))
  else
    return vcat(vcat(ancestor,dw1),vcat(vcat(0,dw2),0))
  end
end

function dyck_word2dyck_path(dw)
  Y=copy(dw)
  for i in 2:size(Y)[1]
    Y[i]+=Y[i-1]
  end
  return Y
end

######## example 
"""
dw1,paths1=tree2dyckword(tree1.root.childs[1][1].childs[1][1],Dict([]))
sort_dw1=triFusion_dw(tree1.root.childs[1][1].childs[1][1],dw1,paths1)
Y1=copy(sort_dw1)
for i in 2:size(Y1)[1]
    Y1[i]+=Y1[i-1]
end"""