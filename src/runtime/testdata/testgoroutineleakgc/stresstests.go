package main

import (
	"os"
	"runtime"
	"runtime/pprof"
	"sync"
)

const spawnGCMaxDepth = 5

func init() {
	register("Spawn", SpawnGC)
}

func spawnGC(i int) {
	prof := pprof.Lookup("goroutineleak")
	if i == 0 {
		return
	}
	wg := &sync.WaitGroup{}
	wg.Add(i + 1)
	go func() {
		wg.Done()
		// deadlocks: x > 0
		<-make(chan int)
	}()
	for j := 0; j < i; j++ {
		go func() {
			wg.Done()
			spawnGC(i - 1)
		}()
	}
	wg.Wait()
	runtime.Gosched()
	prof.WriteTo(os.Stdout, 2)
}

// SpawnGC spawns a tree of goroutine leaks and calls the goroutine leak profiler
// for each node in the tree. It is supposed to stress the goroutine leak profiler
// under a heavily concurrent workload.
func SpawnGC() {
	spawnGC(spawnGCMaxDepth)
}
