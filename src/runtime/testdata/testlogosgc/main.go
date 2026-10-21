// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Command testlogosgc is the helper program of the M19-S1 collector mode
// attacks (src/runtime/logos_gcmode_test.go). Each argument selects one
// scenario and the program prints its observations on standard output.
package main

import (
	"fmt"
	"os"
	"runtime"
	"runtime/debug"
	"strconv"
	"sync"
)

type node struct {
	id    int
	left  *node
	peer  *node
	kids  []*node
	extra [2]*node
}

var sink []byte

// storePtr stores one heap pointer through another; the codegen attack reads
// its assembly for write barrier instructions.
//
//go:noinline
func storePtr(dst **node, v *node) { *dst = v }

func memStats() runtime.MemStats {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return m
}

func churn(mib int) {
	for range mib {
		sink = make([]byte, 1<<20)
	}
}

// epsilon allocates 120 MiB with GOMEMLIMIT unset, raises GOGC, allocates
// more, and finally calls runtime.GC; it reports NumGC after each step.
func epsilon() {
	churn(120)
	idle := memStats().NumGC
	debug.SetGCPercent(1)
	churn(64)
	raised := memStats().NumGC
	runtime.GC()
	fmt.Printf("numgc=%d raised=%d explicit=%d\n", idle, raised, memStats().NumGC)
}

// batch allocates garbage until heap use reaches 80% of the memory limit and
// then until a collection starts. Each phase stops after allocating twice the
// limit, so a collector that keeps the heap small ends the first phase too.
func batch() {
	limit := uint64(debug.SetMemoryLimit(-1))
	budget := limit >> 20 * 2
	m := memStats()
	for i := uint64(0); i < budget && m.HeapAlloc < limit/10*8; i++ {
		sink = make([]byte, 1<<20)
		m = memStats()
	}
	fmt.Printf("eighty numgc=%d heap=%d\n", m.NumGC, m.HeapAlloc)
	first := m.NumGC
	prev := m.HeapAlloc
	for i := uint64(0); i < budget && m.NumGC == first; i++ {
		prev = m.HeapAlloc
		sink = make([]byte, 1<<20)
		m = memStats()
	}
	fmt.Printf("trigger numgc=%d before=%d limit=%d\n", m.NumGC, prev, limit)
}

// graph builds a pointer graph with plain stores from several goroutines,
// lets the collector run under the memory limit and then explicitly, and
// verifies every link.
func graph(rounds int) {
	const n, workers = 20000, 4
	nodes := make([]*node, n)
	byPtr := make(map[*node]int, n)
	for i := range nodes {
		nodes[i] = &node{id: i}
		byPtr[nodes[i]] = i
	}
	for i, nd := range nodes {
		nd.peer = nodes[(i*7+3)%n]
	}
	var wg sync.WaitGroup
	for w := range workers {
		wg.Go(func() {
			for round := range rounds {
				for i := w; i < n; i += workers {
					nd := nodes[i]
					kids := make([]*node, 1+round%3)
					for k := range kids {
						kids[k] = &node{id: i + (k+1)*n, peer: nd}
					}
					storePtr(&nd.left, kids[0])
					nd.kids = kids
					nd.extra[round&1] = kids[0]
				}
			}
		})
	}
	wg.Wait()
	runtime.GC()
	runtime.GC()
	last := 1 + (rounds-1)%3
	for i, nd := range nodes {
		switch {
		case nd.id != i, nd.peer != nodes[(i*7+3)%n], byPtr[nd] != i:
			fail("node %d: id=%d peer=%v index=%d", i, nd.id, nd.peer != nodes[(i*7+3)%n], byPtr[nd])
		case len(nd.kids) != last, nd.left != nd.kids[0], nd.left.id != i+n, nd.left.peer != nd:
			fail("node %d: kids=%d left=%v", i, len(nd.kids), nd.left != nd.kids[0])
		}
	}
	fmt.Printf("ok numgc=%d\n", memStats().NumGC)
}

func fail(format string, args ...any) {
	fmt.Printf("corrupt: "+format+"\n", args...)
	os.Exit(1)
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: testlogosgc epsilon|batch|graph ROUNDS")
		os.Exit(2)
	}
	switch os.Args[1] {
	case "epsilon":
		epsilon()
	case "batch":
		batch()
	case "graph":
		rounds, err := strconv.Atoi(os.Args[2])
		if err != nil || rounds <= 0 {
			fmt.Fprintln(os.Stderr, "bad ROUNDS")
			os.Exit(2)
		}
		graph(rounds)
	default:
		fmt.Fprintln(os.Stderr, "unknown scenario", os.Args[1])
		os.Exit(2)
	}
}
