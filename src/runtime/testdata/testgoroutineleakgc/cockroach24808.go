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

func NewStopper_cockroach24808() *Stopper_cockroach24808 {
	s := &Stopper_cockroach24808{
		stopper: make(chan struct{}),
	}
	return s
}

func NewCompactor_cockroach24808() *Compactor_cockroach24808 {
	return &Compactor_cockroach24808{ch: make(chan struct{}, 1)}
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
	go func() {
		// deadlocks: 1
		stopper := NewStopper_cockroach24808()
		defer stopper.Stop()

		compactor := NewCompactor_cockroach24808()
		compactor.ch <- struct{}{}

		compactor.Start(context.Background(), stopper)
	}()
}
