
declare -a agg_models=("linear" "icfwG" "icfwLA" "icfwGA")
declare -a sim_models=("manhattanSim" "chi2Sim" "cosineSim")

for agg in "${agg_models[@]}"; do
    for sim in "${sim_models[@]}"; do
        echo "python evaluate.py -model_name $agg -simModelName $sim -index_name imdb";
        python evaluate.py -model_name $agg -simModelName $sim -index_name imdb 
    done
done
