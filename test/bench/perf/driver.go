package main

import (
	"flag"
	"fmt"
	"log"
	"time"
	"runtime"
)

var (
	benchNum  = flag.Int("benchnum", 3, "run each benchmark that many times")
	benchTime = flag.Duration("benchtime", 10*time.Second, "benchmarking time for a single run")
	benchMem  = flag.Int("benchmem", 64, "approx RSS value to aim at in benchmarks, in MB")
)

type PerfResult struct {
	N       int64
	RunTime time.Duration
	Metrics []PerfMetric
}

type PerfMetric struct {
	Type  string
	Val   int64
}

type BenchFunc func(N int64) ([]PerfMetric, error)

func PerfBenchmark(f BenchFunc) {
	if !flag.Parsed() {
		flag.Parse()
	}
	var res PerfResult
	for i := 0; i < *benchNum; i++ {
		res1 := RunBenchmark(f)
		if res.RunTime == 0 || res.RunTime > res1.RunTime {
			res = res1
		}
	}
	fmt.Printf("GOPERF-METRIC:runtime=%v\n", int64(res.RunTime)/res.N)
	for _, m := range res.Metrics {
		fmt.Printf("GOPERF-METRIC:%v=%v\n", m.Type, m.Val)
	}
}

func RunBenchmark(f BenchFunc) PerfResult {
	var res PerfResult
	for ChooseN(&res) {
		log.Printf("Benchmarking %v iterations\n", res.N)
		res = RunOnce(f, res.N)
		log.Printf("Done: %+v\n", res)
	}
	return res
}

func RunOnce(f BenchFunc, N int64) PerfResult {
	runtime.GC()
	mstats0 := new(runtime.MemStats)
	runtime.ReadMemStats(mstats0)
	res := PerfResult{N: N}

	t0 := time.Now()
	var err error
	res.Metrics, err = f(N)
	res.RunTime = time.Since(t0)

	if err != nil {
		log.Fatalf("Benchmark function failed: %v\n", err)
	}

	mstats1 := new(runtime.MemStats)
	runtime.ReadMemStats(mstats1)
	fmt.Printf("%+v\n", *mstats1)
	return res
}

func ChooseN(res *PerfResult) bool {
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
