declare -A model_list1
declare -A model_list2
declare -A categ_list
categ_list=(["vanilla"]="" ["vanilla_reverse"]="reverse_")
dataset_list1="mnist"
model_list1=(["mnist_0"]="mnist_0" ["mnist_03"]="mnist_03")
dataset_list2="mnist cifar"
model_list2=(["2-layer"]="mlp" ["normal"]="cnn" ["distilled"]="dd")
lines=1
curtime=`date +"%Y-%m-%d-%I-%M-%p"`
numimg=20
tempfname="tempfile_`echo $curtime`.txt"

for categ in ${!categ_list[@]}; do
	for dataset in $dataset_list1; do
		for model in ${!model_list1[@]}; do
			python3 collect_gradients.py --numimg $numimg --target_type 16 --dataset $dataset --model_name $model --save ./lipschitz_mat/$categ --ids cknn_spade_nodelist/`echo ${model_list1[$model]}`_`echo ${categ_list[$categ]}`TopNode.csv
		done
	done

	for dataset in $dataset_list2; do
		for model in ${!model_list2[@]}; do
			python3 collect_gradients.py --numimg $numimg --target_type 16 --dataset $dataset --model_name $model --save ./lipschitz_mat/$categ --ids cknn_spade_nodelist/`echo $dataset`_`echo ${model_list2[$model]}`_`echo ${categ_list[$categ]}`TopNode.csv
		done
	done
done


echo "">$tempfname
for categ in ${!categ_list[@]}; do
	fname=clever_score/clever_score_`echo $categ`_num`echo $numimg`_`echo $curtime`.txt
	echo "">$fname
	for dataset in $dataset_list1; do
		for model in ${!model_list1[@]}; do
			echo $dataset $categ $model >> $fname
			python3 clever.py lipschitz_mat/$categ/`echo $dataset`_`echo $model` > $tempfname
			echo "$(tail -$lines $tempfname)">>$fname
			# echo lipschitz_mat/$categ/`echo $dataset`_`echo $model`
		done
	done
	
	for dataset in $dataset_list2; do
		for model in ${!model_list2[@]}; do
			echo $dataset $categ $model >> $fname
			python3 clever.py lipschitz_mat/$categ/`echo $dataset`_`echo $model` > $tempfname
			echo "$(tail -$lines $tempfname)">>$fname
			# echo lipschitz_mat/$categ/`echo $dataset`_`echo $model`
		done
	done
done

rm $tempfname