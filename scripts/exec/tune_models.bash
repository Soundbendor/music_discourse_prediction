for model in ada knn lightgbm mlp rf svm; do
    prediction -c "etc/experiments/regression_experiment_${model}.ini" "etc/features/moaf/AMG1608_MOAF_${model}.csv" -o "AMG1608_MOAF.csv" 
done