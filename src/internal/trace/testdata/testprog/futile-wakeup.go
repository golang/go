// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests to make sure the runtime doesn't generate futile wakeups. For example,
// it makes sure that a block on a channel send that unblocks briefly only to
// immediately go back to sleep (in such a way that doesn't reveal any useful
// information, and is purely an artifact of the runtime implementation) doesn't
// make it into the trace.

//go:build ignore

package main

import (
	"context"
	"log"
	"os"
	"runtime"
	"runtime/trace"
	"sync"
)

func main() {
	if err := trace.Start(os.Stdout); err != nil {
		log.Fatalf("failed to start tracing: %v", err)
	}

	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(8))
	c0 := make(chan int, 1)
	c1 := make(chan int, 1)
	c2 := make(chan int, 1)
	const procs = 2
	var done sync.WaitGroup
	done.Add(4 * procs)
	for p := 0; p < procs; p++ {
		const iters = 1e3
		go func() {
			trace.WithRegion(context.Background(), "special", func() {
				for i := 0; i < iters; i++ {
					runtime.Gosched()
					c0 <- 0
				}
				done.Done()
			})
		}()
		go func() {
			trace.WithRegion(context.Background(), "special", func() {
				for i := 0; i < iters; i++ {
					runtime.Gosched()
					<-c0
				}
				done.Done()
			})
		}()
		go func() {
			trace.WithRegion(context.Background(), "special", func() {
				for i := 0; i < iters; i++ {
					runtime.Gosched()
					select {
					case c1 <- 0:
					case c2 <- 0:
					}
				}
				done.Done()
			})
		}()
		go func() {
			trace.WithRegion(context.Background(), "special", func() {
				for i := 0; i < iters; i++ {
					runtime.Gosched()
					select {
					case <-c1:
					case <-c2:
					}
				}
				done.Done()
			})
		}()
	}
	done.Wait()

	trace.Stop()
}
