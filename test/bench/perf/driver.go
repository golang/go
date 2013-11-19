package main

import (
	"flag"
	"fmt"
	"os"
	"time"
)

var (
	benchNum  = flag.Int("benchnum", 3, "run each benchmark that many times")
	benchTime = flag.Duration("benchtime", 10*time.Second, "benchmarking time for a single run")
	benchMem  = flag.Int("benchmem", 64, "approx RSS value to aim at in benchmarks, in MB")
)

type Result struct {
	N       int64
	RunTime time.Duration
}

func main() {
	flag.Parse()
	var res Result
	for i := 0; i < *benchNum; i++ {
		res1 := RunBenchmark()
		if res.RunTime == 0 || res.RunTime > res1.RunTime {
			res = res1
		}
	}
	fmt.Printf("GOPERF-METRIC:runtime=%v\n", int64(res.RunTime)/res.N)
}

func RunBenchmark() Result {
	var res Result
	for ChooseN(&res) {
		res = RunOnce(res.N)
	}
	return res
}

func RunOnce(N int64) Result {
	fmt.Printf("Benchmarking %v iterations\n", N)
	t0 := time.Now()
	err := Benchmark(N)
	if err != nil {
		fmt.Printf("Benchmark function failed: %v\n", err)
		os.Exit(1)
	}
	res := Result{N: N}
	res.RunTime = time.Since(t0)
	return res
}

func ChooseN(res *Result) bool {
	const MaxN = 1e12
	last := res.N
	if last == 0 {
		res.N = 1
		return true
	} else if res.RunTime >= *benchTime || last >= MaxN {
		return false
	}
	nsPerOp := max(1, int64(res.RunTime)/last)
	res.N = int64(*benchTime) / nsPerOp
	res.N = max(min(res.N+res.N/2, 100*last), last+1)
	res.N = roundUp(res.N)
	return true
}

func roundUp(n int64) int64 {
	tmp := n
	base := int64(1)
	for tmp >= 10 {
		tmp /= 10
		base *= 10
	}
	switch {
	case n <= base:
		return base
	case n <= (2 * base):
		return 2 * base
	case n <= (5 * base):
		return 5 * base
	default:
		return 10 * base
	}
	panic("unreachable")
	return 0
}

func min(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

func max(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}
