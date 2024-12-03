gpt2-baseline:
	gcc -o output gpt2.c -lm
	./output

gpt2-optimized:
	gcc -fopenmp -O3 -funroll-loops -march=native -flto -o gptop gpt2-optimized.c -lm
	./gptop

clean:
	rm -f gptop
	rm -f output
