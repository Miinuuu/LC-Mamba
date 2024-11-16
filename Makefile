train:
	python -m torch.distributed.launch --nproc_per_node=2 train.py  --model csdw_dec_dq --world_size 2 --batch_size 16 --data_path /data/datasets/vimeo_triplet

bench_Ours-E:
	CUDA_VISIBLE_DEVICES=0  python model_benchmark.py --model Ours-E --resume Ours-E --bench [Vimeo90K]

bench_Ours-B:
	CUDA_VISIBLE_DEVICES=0  python model_benchmark.py --model Ours-E --resume Ours-B --bench [Vimeo90K]

bench_Ours-P:
	CUDA_VISIBLE_DEVICES=0  python model_benchmark.py --model Ours-E --resume Ours-P --bench [Vimeo90K]

demo :
	python demo.py --model csdw_dec_dq  --resume csdw_dec_dq_299_36.14  
