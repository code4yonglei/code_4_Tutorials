#!/bin/bash

inod=g4
for imol in HBD IM
do
    mkdir $imol
    cd    $imol

    mv ../$imol.gjf .
    cp ../*.pbs .

    sed -i -- 's+xmolx+'$imol'+g' *.pbs
    sed -i -- 's+gnode+'$inod'+g' *.pbs
    current_path=$(pwd)
    sed -i -- 's+mmk_path+'$current_path'+g' 01_mmk_gauss.pbs

    qsub 01_mmk_gauss.pbs

    sleep 2s
    cd ../

done
