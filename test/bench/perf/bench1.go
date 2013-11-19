package main

import (
	"time"
)

func Benchmark(N int64) error {
	// 13+
	time.Sleep(time.Duration(N) * time.Millisecond)
	return nil
}
