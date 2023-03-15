include("/home/chirraneso/training/simulator_DL/simulator.jl")
include("/home/chirraneso/training/phylogeny_simu_abc.jl")
include("/home/chirraneso/training/tree2adjmatrix.jl")

function prior(n)
    theta=Vector{Any}(nothing,n)
    for i in 1:n
        y = rand(Uniform(0,0.5)) #p0
        x = rand(Uniform(0,1)) #p2
        while x+y > 1 || y >= x #|| y/x > 0.90
        y = rand(Uniform(0,0.5)) #p0
        x = rand(Uniform(0,1)) #p2
        end
        theta[i]=[x,y]
    end
    return theta#Penser à avoir le même ordre que le modèle dummy
end

N = 10
nb_leafs = 20
t_max = 300

dataset = Dict()
dataset["p0"]=Vector{Any}(nothing,10*N)
dataset["p2"]=Vector{Any}(nothing,10*N)
dataset["delta"] =Vector{Any}(nothing,10*N)
dataset["q"] = Vector{Any}(nothing,10*N)
dataset["q_class"]=Vector{Any}(nothing,10*N)
dataset["delta_class"] = Vector{Any}(nothing,10*N)
dataset["coo"] = Vector{Any}(nothing,10*N)
dataset["node_features"] = Vector{Any}(nothing,10*N)
dataset["rscore"] = Vector{Any}(nothing,10*N)
dataset["dyck"] = Vector{Any}(nothing,10*N)
dataset["LTT"] = Vector{Any}(nothing,10*N)

param = prior(N)

for i in 1:N
    p2,p0 = param[i][1], param[i][2]
    for j in 1:10
        dict = dico_simu(p0,p2, nb_leafs,t_max)
        for k in keys(dataset)
            dataset[k][i] = dict[k]
        end
    end
end
    
using JSON
str = JSON.json(dataset)
write("dataset10.txt", str)


"""str = read("data.txt", String)
data = JSON.parse(str)"""

