train:
	python -m torch.distributed.launch --nproc_per_node=2 train.py  --model csdw_dec_dq --world_size 2 --batch_size 16 --data_path /data/datasets/vimeo_triplet

bench:
	CUDA_VISIBLE_DEVICES=0  python model_benchmark.py --model csdw_dec_dq --resume csdw_dec_dq_299_36.14 --bench [Vimeo90K,UCF101,SNU_FILM,MiddleBury]

demo :
	python demo.py --model csdw_dec_dq  --resume csdw_dec_dq_299_36.14  
