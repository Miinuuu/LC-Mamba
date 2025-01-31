bench_Ours-E:
	CUDA_VISIBLE_DEVICES=1  python model_benchmark.py --model Ours-E --resume Ours-E 

bench_Ours-B:
	CUDA_VISIBLE_DEVICES=1  python model_benchmark.py --model Ours-B --resume Ours-B 

bench_Ours-P:
	CUDA_VISIBLE_DEVICES=0  python model_benchmark.py --model Ours-P --resume Ours-P

bench_Ours-P+S:
	CUDA_VISIBLE_DEVICES=0  python model_benchmark.py --model Ours-P --resume Ours-P+S
