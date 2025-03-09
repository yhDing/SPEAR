# 默认设备ID为0
DEVICE_ID=0

# 解析命令行参数
for arg in "$@"; do
    if [[ $arg == --device_id=* ]]; then
        DEVICE_ID="${arg#*=}"
    fi
done


#Cora
#none
python -u main.py --dataset=Cora --homo_loss_weight=0.1 --target_loss_weight=1 --vs_number=10 --test_model=GCN --defense_mode=none --epochs=200 --trojan_epochs=200 --device_id=$DEVICE_ID  --alpha_int=30 --hidden=80 --shadow_lr=0.0002 --trojan_lr=0.0002
#prune
python -u main.py --prune_thr=0.1 --dataset=Cora --homo_loss_weight=1 --target_loss_weight=1 --vs_number=10 --test_model=GCN --defense_mode=prune --epochs=200 --trojan_epochs=200 --device_id=$DEVICE_ID  --alpha_int=30 --hidden=80 --shadow_lr=0.0003 --trojan_lr=0.0003
#od
python -u main.py --dataset=Cora --homo_loss_weight=5 --target_loss_weight=4 --vs_number=10 --test_model=GCN --defense_mode=reconstruct --epochs=200 --trojan_epochs=200 --device_id=$DEVICE_ID  --alpha_int=30 --hidden=80 --shadow_lr=0.0003 --trojan_lr=0.0003
#rigbd
python -u main_rigbd.py --dataset=Cora --homo_loss_weight=0.1 --target_loss_weight=1 --vs_number=10 --test_model=GCN --defense_mode=none --epochs=200 --trojan_epochs=200 --device_id=$DEVICE_ID  --alpha_int=30 --hidden=80 --shadow_lr=0.0002 --trojan_lr=0.0002
#Pubmed
#none
python -u main.py --dataset=Pubmed --homo_loss_weight=0.1 --vs_number=40 --test_model=GCN --defense_mode=none --epochs=200 --trojan_epochs=400 --device_id=$DEVICE_ID  --alpha_int=10 --hidden=64 --target_class=2 --shadow_lr=0.01 --trojan_lr=0.01
#prune
python -u main.py --prune_thr=0.2 --dataset=Pubmed --homo_loss_weight=2 --vs_number=40 --test_model=GCN --defense_mode=prune --epochs=200 --trojan_epochs=400 --device_id=$DEVICE_ID  --alpha_int=10 --hidden=64 --target_class=2 --shadow_lr=0.01 --trojan_lr=0.01
#od
python -u main.py --dataset=Pubmed --homo_loss_weight=1 --target_loss_weight=1 --vs_number=157 --test_model=GCN --defense_mode=reconstruct --epochs=200 --trojan_epochs=500 --device_id=$DEVICE_ID  --alpha_int=10 --hidden=64 --target_class=2 --shadow_lr=0.01 --trojan_lr=0.01
#rigbd
python -u main_rigbd.py --dataset=Pubmed --homo_loss_weight=1 --vs_number=40 --test_model=GCN --epochs=200 --trojan_epochs=400 --train_lr=0.01 --device_id=$DEVICE_ID  --alpha_int=10 --hidden=64 --target_class=2 --shadow_lr=0.01 --trojan_lr=0.01
#OGB-arxiv
#none
python -u main.py --dataset=ogbn-arxiv --homo_loss_weight=0 --vs_number=565 --test_model=GCN --defense_mode=none --epochs=800 --trojan_epochs=800 --device_id=$DEVICE_ID  --alpha_int=5 --hidden=80 --outter_size=256 --shadow_lr=0.001 --trojan_lr=0.001
#prune
python -u main.py --dataset=ogbn-arxiv --homo_loss_weight=4 --vs_number=565 --test_model=GCN --defense_mode=prune --epochs=800 --trojan_epochs=800 --device_id=$DEVICE_ID  --alpha_int=5 --hidden=80 --outter_size=256 --shadow_lr=0.001 --trojan_lr=0.001
#od
python -u main.py --dataset=ogbn-arxiv --homo_loss_weight=10 --vs_number=1693 --test_model=GCN --defense_mode=reconstruct --epochs=800 --trojan_epochs=600 --device_id=$DEVICE_ID  --alpha_int=5 --hidden=64 --outter_size=512 --shadow_lr=0.001 --trojan_lr=0.001
#rigbd
python -u main_rigbd.py --dataset=ogbn-arxiv --homo_loss_weight=0 --vs_number=565 --test_model=GCN --defense_mode=none --epochs=800 --trojan_epochs=800 --device_id=$DEVICE_ID  --alpha_int=5 --hidden=80 --outter_size=256 --shadow_lr=0.001 --trojan_lr=0.001
#GAT--Need at least 25G GPU memory
python -u main.py --dataset=ogbn-arxiv --homo_loss_weight=0 --vs_number=565 --test_model=GAT --epochs=800 --train_lr=0.01 --trojan_epochs=800 --device_id=0 --alpha_int=5 --hidden=80 --outter_size=256 --shadow_lr=0.001 --trojan_lr=0.001
#GrapgSAGE
python -u main.py --dataset=ogbn-arxiv --homo_loss_weight=5 --vs_number=565 --test_model=GraphSage --epochs=800 --train_lr=0.01 --trojan_epochs=800 --device_id=$DEVICE_ID  --alpha_int=5 --hidden=80 --outter_size=256 --shadow_lr=0.001 --trojan_lr=0.001
#GNNGuard
python -u main.py --dataset=ogbn-arxiv --homo_loss_weight=5 --vs_number=565 --test_model=GNNGuard --epochs=800 --train_lr=0.01 --trojan_epochs=800 --device_id=$DEVICE_ID  --alpha_int=5 --hidden=80 --outter_size=256 --shadow_lr=0.001 --trojan_lr=0.001
#RobustGCN--Take several hours on 4090.
python -u main.py --dataset=ogbn-arxiv --homo_loss_weight=5 --vs_number=565 --test_model=RobustGCN --epochs=800 --train_lr=0.01 --trojan_epochs=800 --device_id=$DEVICE_ID  --alpha_int=5 --hidden=80 --outter_size=256 --shadow_lr=0.001 --trojan_lr=0.001
