# Round-2 compute benchmark and concurrency choice

Before final execution, one representative development S1 selection task ran with two capped
threads in 17.42 seconds wall time, using 27.00 seconds user CPU, 0.77 seconds system CPU, 159% CPU,
and 755,424 KiB maximum resident memory. The host has 48 logical CPUs and 503 GiB RAM.

After the source checkpoint, one actual final n=1500, 300-epoch, fully nested selection task ran
in 103.77 seconds wall time, used 199.15 seconds user CPU (196%), and reached 791,980 KiB maximum
RSS. It completed successfully and recorded checkpoint `aa983a2`.

The bounded local launcher is therefore frozen at **12 workers × 2 threads** for final selection
jobs. This uses at most 24 logical CPUs and about 9.5 GiB measured aggregate task memory, leaving
ample headroom.
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and PyTorch
CPU threads are explicitly capped. A2 is cheaper but uses the same conservative launcher setting.
