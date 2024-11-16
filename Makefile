bench_Ours-E:
	CUDA_VISIBLE_DEVICES=0  python model_benchmark.py --model Ours-E --resume Ours-E 

bench_Ours-B:
	CUDA_VISIBLE_DEVICES=0  python model_benchmark.py --model Ours-E --resume Ours-B 

bench_Ours-P:
	CUDA_VISIBLE_DEVICES=0  python model_benchmark.py --model Ours-E --resume Ours-P 

