#bash

num=10
name=simple


for i in `seq 1 1 $num`; do

echo $starting simulation run $i
python -m langevin_model.simulation sim --name $name --steps 10000000
python -m langevin_model.simulation jac --name $name --target ideal_set.dat
python -m langevin_model.simulation fit --name $name
python -m langevin_model.simulation next --name $name --start

done

echo $starting final run
python -m langevin_model.simulation sim --name $name --steps 10000000
python -m langevin_model.simulation jac --name $name --target ideal_set.dat
python -m langevin_model.simulation fit --name $name
python -m langevin_model.simulation next --name $name
