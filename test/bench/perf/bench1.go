package main

import (
	"time"
)

func main() {
	PerfBenchmark(SleepBenchmark)
}

func SleepBenchmark(N int64) (metrics []PerfMetric, err error) {
	time.Sleep(time.Duration(N) * time.Millisecond)
	metrics = append(metrics, PerfMetric{"foo", 42})
	return
}
