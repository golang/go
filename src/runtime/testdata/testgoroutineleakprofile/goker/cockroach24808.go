// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

package main

import (
	"context"
	"os"
	"runtime"
	"runtime/pprof"
	"sync"
)

func init() {
	register("Cockroach24808", Cockroach24808)
}

type Compactor_cockroach24808 struct {
	ch chan struct{}
}

type Stopper_cockroach24808 struct {
	stop    sync.WaitGroup
	stopper chan struct{}
}

func (s *Stopper_cockroach24808) RunWorker(ctx context.Context, f func(context.Context)) {
	s.stop.Add(1)
	go func() {
		defer s.stop.Done()
		f(ctx)
	}()
}

func (s *Stopper_cockroach24808) ShouldStop() <-chan struct{} {
	if s == nil {
		return nil
	}
	return s.stopper
}

func (s *Stopper_cockroach24808) Stop() {
	close(s.stopper)
}

func (c *Compactor_cockroach24808) Start(ctx context.Context, stopper *Stopper_cockroach24808) {
	c.ch <- struct{}{}
	stopper.RunWorker(ctx, func(ctx context.Context) {
		for {
			select {
			case <-stopper.ShouldStop():
				return
			case <-c.ch:
			}
		}
	})
}

func Cockroach24808() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		// Yield several times to allow the child goroutine to run.
		for i := 0; i < yieldCount; i++ {
			runtime.Gosched()
		}
		prof.WriteTo(os.Stdout, 2)
	}()
	go func() { // G1
		stopper := &Stopper_cockroach24808{stopper: make(chan struct{})}
		defer stopper.Stop()

		compactor := &Compactor_cockroach24808{ch: make(chan struct{}, 1)}
		compactor.ch <- struct{}{}

		compactor.Start(context.Background(), stopper)
	}()
}
