#!/bin/bash

vocab=$PWD/minmt-vocab.py
setup=$PWD/minmt-setup.py
train=$PWD/minmt-train.py
avrge=$PWD/minmt-average.py
trans=$PWD/minmt-translate.py

tokenizer=$PWD/tools/tokenizer.py

josep=/nfs/RESEARCH/crego/projects/PrimingNMT-2/
data=$josep/data
stovec=$josep/stovec


dir=$josep/minmt_base
dnet=$PWD/model_base_dan

# toutes les lignes suivantes sont pour construire des fichiers déjà construits
  #cat $data/*.??.trn.bpe | python3 $vocab -max_size 32000 > $dir/enfr.BPE.32k.voc
	#rm -f $dir/{trn,val}.{en,fr}
	#cat $data/*.en.trn.bpe > $dir/trn.en
	#cat $data/*.fr.trn.bpe > $dir/trn.fr
	#cat $data/*.en.val.bpe > $dir/val.en
	#cat $data/*.fr.val.bpe > $dir/val.fr

	python3 $setup -dnet $dnet -src_voc $dir/enfr.BPE.32k.voc -tgt_voc $dir/enfr.BPE.32k.voc

	CUDA_VISIBLE_DEVICES=0 python3 $train -dnet $dnet -src_train $dir/trn.en -tgt_train $dir/trn.fr -src_valid $dir/val.en -tgt_valid $dir/val.fr -max_steps 450000 -loss KLDiv -cuda -log_file $dnet/log &

echo fin du training


