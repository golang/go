// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
	"runtime"
	"runtime/metrics"
	"sync"
	"sync/atomic"
	"time"
)

func init() {
	register("LogosPacerElastic", LogosPacerElastic)
	register("LogosPacerAssist", LogosPacerAssist)
}

type logosPacerNode struct {
	next *logosPacerNode
	pad  [5]*logosPacerNode
}

var (
	logosPacerLive *logosPacerNode
	logosPacerSink *logosPacerNode
)

func LogosPacerElastic() {
	runtime.GOMAXPROCS(4)
	nodes, cycles := 1<<20, uint64(6)
	if os.Getenv("RUNTIME_TEST_SHORT") != "" {
		nodes, cycles = 1<<19, 3
	}
	for range nodes {
		logosPacerLive = &logosPacerNode{next: logosPacerLive}
	}
	sample := []metrics.Sample{{Name: "/gc/cycles/total:gc-cycles"}}
	collections := func() uint64 {
		metrics.Read(sample)
		return sample[0].Value.Uint64()
	}
	want := collections() + cycles
	deadline := time.Now().Add(60 * time.Second)
	for collections() < want && time.Now().Before(deadline) {
		for range 1000 {
			logosPacerSink = &logosPacerNode{}
		}
	}
	if collections() < want {
		fmt.Println("FAIL: too few collections")
		return
	}
	fmt.Println("OK")
}

func LogosPacerAssist() {
	runtime.GOMAXPROCS(2)
	nodes := 1 << 19
	if os.Getenv("RUNTIME_TEST_SHORT") != "" {
		nodes = 1 << 18
	}
	for range nodes {
		logosPacerLive = &logosPacerNode{next: logosPacerLive}
	}
	var stop atomic.Bool
	var wg sync.WaitGroup
	sinks := make([]*logosPacerNode, 8)
	for i := range sinks {
		wg.Go(func() {
			for !stop.Load() {
				sinks[i] = &logosPacerNode{}
			}
		})
	}
	sample := []metrics.Sample{{Name: "/cpu/classes/gc/mark/assist:cpu-seconds"}}
	deadline := time.Now().Add(30 * time.Second)
	assisted := false
	for !assisted && time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
		metrics.Read(sample)
		assisted = sample[0].Value.Float64() > 0
	}
	stop.Store(true)
	wg.Wait()
	if !assisted {
		fmt.Println("FAIL: no mark assist ran")
		return
	}
	fmt.Println("OK")
}
