// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests coroutine switches.

//go:build ignore

package main

import (
	"iter"
	"log"
	"os"
	"runtime/trace"
	"sync"
)

func main() {
	// Start tracing.
	if err := trace.Start(os.Stdout); err != nil {
		log.Fatalf("failed to start tracing: %v", err)
	}

	// Try simple pull iteration.
	i := pullRange(100)
	for {
		_, ok := i.next()
		if !ok {
			break
		}
	}

	// Try bouncing the pull iterator between two goroutines.
	var wg sync.WaitGroup
	var iterChans [2]chan intIter
	wg.Add(2)
	iterChans[0] = make(chan intIter)
	iterChans[1] = make(chan intIter)
	go func() {
		defer wg.Done()

		iter := pullRange(100)
		iterChans[1] <- iter

		for i := range iterChans[0] {
			_, ok := i.next()
			if !ok {
				close(iterChans[1])
				break
			}
			iterChans[1] <- i
		}
	}()
	go func() {
		defer wg.Done()

		for i := range iterChans[1] {
			_, ok := i.next()
			if !ok {
				close(iterChans[0])
				break
			}
			iterChans[0] <- i
		}
	}()
	wg.Wait()

	// End of traced execution.
	trace.Stop()
}

func pullRange(n int) intIter {
	next, stop := iter.Pull(func(yield func(v int) bool) {
		for i := range n {
			yield(i)
		}
	})
	return intIter{next: next, stop: stop}
}

type intIter struct {
	next func() (int, bool)
	stop func()
}
