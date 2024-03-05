// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests a GC-heavy program. This is useful for shaking out
// all sorts of corner cases about GC-related ranges.

//go:build ignore

package main

import (
	"log"
	"math/rand"
	"os"
	"runtime"
	"runtime/trace"
	"time"
)

type node struct {
	children [4]*node
	data     [128]byte
}

func makeTree(depth int) *node {
	if depth == 0 {
		return new(node)
	}
	return &node{
		children: [4]*node{
			makeTree(depth - 1),
			makeTree(depth - 1),
			makeTree(depth - 1),
			makeTree(depth - 1),
		},
	}
}

var trees [16]*node
var ballast *[16]*[8192]*node
var sink [][]byte

func main() {
	for i := range trees {
		trees[i] = makeTree(6)
	}
	ballast = new([16]*[8192]*node)
	for i := range ballast {
		ballast[i] = new([8192]*node)
		for j := range ballast[i] {
			ballast[i][j] = &node{
				data: [128]byte{1, 2, 3, 4},
			}
		}
	}

	procs := runtime.GOMAXPROCS(-1)
	sink = make([][]byte, procs)

	for i := 0; i < procs; i++ {
		i := i
		go func() {
			for {
				sink[i] = make([]byte, rand.Intn(32<<10))
			}
		}()
	}
	// Increase the chance that we end up starting and stopping
	// mid-GC by only starting to trace after a few milliseconds.
	time.Sleep(5 * time.Millisecond)

	// Start tracing.
	if err := trace.Start(os.Stdout); err != nil {
		log.Fatalf("failed to start tracing: %v", err)
	}
	defer trace.Stop()

	// Let the tracing happen for a bit.
	time.Sleep(400 * time.Millisecond)
}
