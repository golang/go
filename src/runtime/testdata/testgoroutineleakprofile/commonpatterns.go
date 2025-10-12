// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"runtime/pprof"
	"time"
)

// Common goroutine leak patterns. Extracted from:
// "Unveiling and Vanquishing Goroutine Leaks in Enterprise Microservices: A Dynamic Analysis Approach"
// doi:10.1109/CGO57630.2024.10444835
//
// Tests in this file are not flaky iff. the test is run with GOMAXPROCS=1.
// The main goroutine forcefully yields via `runtime.Gosched()` before
// running the profiler. This moves them to the back of the run queue,
// allowing the leaky goroutines to be scheduled beforehand and get stuck.

func init() {
	register("NoCloseRange", NoCloseRange)
	register("MethodContractViolation", MethodContractViolation)
	register("DoubleSend", DoubleSend)
	register("EarlyReturn", EarlyReturn)
	register("NCastLeak", NCastLeak)
	register("Timeout", Timeout)
}

// Incoming list of items and the number of workers.
func noCloseRange(list []any, workers int) {
	ch := make(chan any)

	// Create each worker
	for i := 0; i < workers; i++ {
		go func() {

			// Each worker waits for an item and processes it.
			for item := range ch {
				// Process each item
				_ = item
			}
		}()
	}

	// Send each item to one of the workers.
	for _, item := range list {
		// Sending can leak if workers == 0 or if one of the workers panics
		ch <- item
	}
	// The channel is never closed, so workers leak once there are no more
	// items left to process.
}

func NoCloseRange() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()

	go noCloseRange([]any{1, 2, 3}, 0)
	go noCloseRange([]any{1, 2, 3}, 3)
}

// A worker processes items pushed to `ch` one by one in the background.
// When the worker is no longer needed, it must be closed with `Stop`.
//
// Specifications:
//
//	A worker may be started any number of times, but must be stopped only once.
//		Stopping a worker multiple times will lead to a close panic.
//	Any worker that is started must eventually be stopped.
//		Failing to stop a worker results in a goroutine leak
type worker struct {
	ch   chan any
	done chan any
}

// Start spawns a background goroutine that extracts items pushed to the queue.
func (w worker) Start() {
	go func() {

		for {
			select {
			case <-w.ch: // Normal workflow
			case <-w.done:
				return // Shut down
			}
		}
	}()
}

func (w worker) Stop() {
	// Allows goroutine created by Start to terminate
	close(w.done)
}

func (w worker) AddToQueue(item any) {
	w.ch <- item
}

// worker limited in scope by workerLifecycle
func workerLifecycle(items []any) {
	// Create a new worker
	w := worker{
		ch:   make(chan any),
		done: make(chan any),
	}
	// Start worker
	w.Start()

	// Operate on worker
	for _, item := range items {
		w.AddToQueue(item)
	}

	runtime.Gosched()
	// Exits without calling ’Stop’. Goroutine created by `Start` eventually leaks.
}

func MethodContractViolation() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		runtime.Gosched()
		prof.WriteTo(os.Stdout, 2)
	}()

	workerLifecycle(make([]any, 10))
	runtime.Gosched()
}

// doubleSend incoming channel must send a message (incoming error simulates an error generated internally).
func doubleSend(ch chan any, err error) {
	if err != nil {
		// In case of an error, send nil.
		ch <- nil
		// Return is missing here.
	}
	// Otherwise, continue with normal behaviour
	// This send is still executed in the error case, which may lead to a goroutine leak.
	ch <- struct{}{}
}

func DoubleSend() {
	prof := pprof.Lookup("goroutineleak")
	ch := make(chan any)
	defer func() {
		runtime.Gosched()
		prof.WriteTo(os.Stdout, 2)
	}()

	go func() {
		doubleSend(ch, nil)
	}()
	<-ch

	go func() {
		doubleSend(ch, fmt.Errorf("error"))
	}()
	<-ch

	ch1 := make(chan any, 1)
	go func() {
		doubleSend(ch1, fmt.Errorf("error"))
	}()
	<-ch1
}

// earlyReturn demonstrates a common pattern of goroutine leaks.
// A return statement interrupts the evaluation of the parent goroutine before it can consume a message.
// Incoming error simulates an error produced internally.
func earlyReturn(err error) {
	// Create a synchronous channel
	ch := make(chan any)

	go func() {

		// Send something to the channel.
		// Leaks if the parent goroutine terminates early.
		ch <- struct{}{}
	}()

	if err != nil {
		// Interrupt evaluation of parent early in case of error.
		// Sender leaks.
		return
	}

	// Only receive if there is no error.
	<-ch
}

func EarlyReturn() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		runtime.Gosched()
		prof.WriteTo(os.Stdout, 2)
	}()

	go earlyReturn(fmt.Errorf("error"))
}

// nCastLeak processes a number of items. First result to pass the post is retrieved from the channel queue.
func nCastLeak(items []any) {
	// Channel is synchronous.
	ch := make(chan any)

	// Iterate over every item
	for range items {
		go func() {

			// Process item and send result to channel
			ch <- struct{}{}
			// Channel is synchronous: only one sender will synchronise
		}()
	}
	// Retrieve first result. All other senders block.
	// Receiver blocks if there are no senders.
	<-ch
}

func NCastLeak() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		for i := 0; i < yieldCount; i++ {
			// Yield enough times  to allow all the leaky goroutines to
			// reach the execution point.
			runtime.Gosched()
		}
		prof.WriteTo(os.Stdout, 2)
	}()

	go func() {
		nCastLeak(nil)
	}()

	go func() {
		nCastLeak(make([]any, 5))
	}()
}

// A context is provided to short-circuit evaluation, leading
// the sender goroutine to leak.
func timeout(ctx context.Context) {
	ch := make(chan any)

	go func() {
		ch <- struct{}{}
	}()

	select {
	case <-ch: // Receive message
		// Sender is released
	case <-ctx.Done(): // Context was cancelled or timed out
		// Sender is leaked
	}
}

func Timeout() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		runtime.Gosched()
		prof.WriteTo(os.Stdout, 2)
	}()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	for i := 0; i < 100; i++ {
		go timeout(ctx)
	}
}
