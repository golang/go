// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

package main

import (
	"os"
	"runtime"
	"runtime/pprof"
	"sync"
	"time"
)

func init() {
	register("Serving2137", Serving2137)
}

type token_serving2137 struct{}

type request_serving2137 struct {
	lock     *sync.Mutex
	accepted chan bool
}

type Breaker_serving2137 struct {
	pendingRequests chan token_serving2137
	activeRequests  chan token_serving2137
}

func (b *Breaker_serving2137) Maybe(thunk func()) bool {
	var t token_serving2137
	select {
	default:
		// Pending request queue is full.  Report failure.
		return false
	case b.pendingRequests <- t:
		// Pending request has capacity.
		// Wait for capacity in the active queue.
		b.activeRequests <- t
		// Defer releasing capacity in the active and pending request queue.
		defer func() {
			<-b.activeRequests
			runtime.Gosched()
			<-b.pendingRequests
		}()
		// Do the thing.
		thunk()
		// Report success
		return true
	}
}

func (b *Breaker_serving2137) concurrentRequest() request_serving2137 {
	r := request_serving2137{lock: &sync.Mutex{}, accepted: make(chan bool, 1)}
	r.lock.Lock()
	var start sync.WaitGroup
	start.Add(1)
	go func() { // G2, G3
		start.Done()
		runtime.Gosched()
		ok := b.Maybe(func() {
			// Will block on locked mutex.
			r.lock.Lock()
			runtime.Gosched()
			r.lock.Unlock()
		})
		r.accepted <- ok
	}()
	start.Wait() // Ensure that the go func has had a chance to execute.
	return r
}

// Perform n requests against the breaker, returning mutexes for each
// request which succeeded, and a slice of bools for all requests.
func (b *Breaker_serving2137) concurrentRequests(n int) []request_serving2137 {
	requests := make([]request_serving2137, n)
	for i := range requests {
		requests[i] = b.concurrentRequest()
	}
	return requests
}

func NewBreaker_serving2137(queueDepth, maxConcurrency int32) *Breaker_serving2137 {
	return &Breaker_serving2137{
		pendingRequests: make(chan token_serving2137, queueDepth+maxConcurrency),
		activeRequests:  make(chan token_serving2137, maxConcurrency),
	}
}

func unlock_serving2137(req request_serving2137) {
	req.lock.Unlock()
	runtime.Gosched()
	// Verify that function has completed
	ok := <-req.accepted
	runtime.Gosched()
	// Requeue for next usage
	req.accepted <- ok
}

func unlockAll_serving2137(requests []request_serving2137) {
	for _, lc := range requests {
		unlock_serving2137(lc)
	}
}

func Serving2137() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()

	for i := 0; i < 1000; i++ {
		go func() {
			b := NewBreaker_serving2137(1, 1)

			locks := b.concurrentRequests(2) // G1
			unlockAll_serving2137(locks)
		}()
	}
}
