// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"runtime"
	"runtime/pprof"
	"unsafe"
)

const BranchingFactor = 4

type Object struct {
	child [BranchingFactor]*Object
}

var (
	cpus       = flag.Int("cpus", 1, "number of cpus to use")
	heapsize   = flag.Int64("heapsize", 100*1024*1024, "size of the heap in bytes")
	cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")

	lastPauseNs uint64 = 0
	lastFree    uint64 = 0
	heap        *Object
	calls       [20]int
	numobjects  int64
)

func buildHeap() {
	objsize := int64(unsafe.Sizeof(Object{}))
	heap, _ = buildTree(float64(objsize), float64(*heapsize), 0)
	fmt.Printf("*** built heap: %.0f MB; (%d objects * %d bytes)\n",
		float64(*heapsize)/1048576, numobjects, objsize)
}

func buildTree(objsize, size float64, depth int) (*Object, float64) {
	calls[depth]++
	x := &Object{}
	numobjects++
	subtreeSize := (size - objsize) / BranchingFactor
	alloc := objsize
	for i := 0; i < BranchingFactor && alloc < size; i++ {
		c, n := buildTree(objsize, subtreeSize, depth+1)
		x.child[i] = c
		alloc += n
	}
	return x, alloc
}

func gc() {
	runtime.GC()
	runtime.UpdateMemStats()
	pause := runtime.MemStats.PauseTotalNs
	inuse := runtime.MemStats.Alloc
	free := runtime.MemStats.TotalAlloc - inuse
	fmt.Printf("gc pause: %8.3f ms; collect: %8.0f MB; heapsize: %8.0f MB\n",
		float64(pause-lastPauseNs)/1e6,
		float64(free-lastFree)/1048576,
		float64(inuse)/1048576)
	lastPauseNs = pause
	lastFree = free
}

func main() {
	flag.Parse()
	buildHeap()
	runtime.GOMAXPROCS(*cpus)
	runtime.UpdateMemStats()
	lastPauseNs = runtime.MemStats.PauseTotalNs
	lastFree = runtime.MemStats.TotalAlloc - runtime.MemStats.Alloc
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}
	for i := 0; i < 10; i++ {
		gc()
	}
}
