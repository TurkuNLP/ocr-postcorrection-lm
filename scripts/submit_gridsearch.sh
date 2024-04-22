mkdir -p eo_gridsearch

for i in 4 8 16
do 
    sbatch -e eo_gridsearch/$i.err -o eo_gridsearch/$i.out
done