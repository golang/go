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
	register("Cockroach1462", Cockroach1462)
}

type Stopper_cockroach1462 struct {
	stopper  chan struct{}
	stopped  chan struct{}
	stop     sync.WaitGroup
	mu       sync.Mutex
	drain    *sync.Cond
	draining bool
	numTasks int
}

func NewStopper_cockroach1462() *Stopper_cockroach1462 {
	s := &Stopper_cockroach1462{
		stopper: make(chan struct{}),
		stopped: make(chan struct{}),
	}
	s.drain = sync.NewCond(&s.mu)
	return s
}

func (s *Stopper_cockroach1462) RunWorker(f func()) {
	s.AddWorker()
	go func() { // G2, G3
		defer s.SetStopped()
		f()
	}()
}

func (s *Stopper_cockroach1462) AddWorker() {
	s.stop.Add(1)
}
func (s *Stopper_cockroach1462) StartTask() bool {
	s.mu.Lock()
	runtime.Gosched()
	defer s.mu.Unlock()
	if s.draining {
		return false
	}
	s.numTasks++
	return true
}

func (s *Stopper_cockroach1462) FinishTask() {
	s.mu.Lock()
	runtime.Gosched()
	defer s.mu.Unlock()
	s.numTasks--
	s.drain.Broadcast()
}
func (s *Stopper_cockroach1462) SetStopped() {
	if s != nil {
		s.stop.Done()
	}
}
func (s *Stopper_cockroach1462) ShouldStop() <-chan struct{} {
	if s == nil {
		return nil
	}
	return s.stopper
}

func (s *Stopper_cockroach1462) Quiesce() {
	s.mu.Lock()
	runtime.Gosched()
	defer s.mu.Unlock()
	s.draining = true
	for s.numTasks > 0 {
		// Unlock s.mu, wait for the signal, and lock s.mu.
		s.drain.Wait()
	}
}

func (s *Stopper_cockroach1462) Stop() {
	s.Quiesce()
	close(s.stopper)
	s.stop.Wait()
	s.mu.Lock()
	runtime.Gosched()
	defer s.mu.Unlock()
	close(s.stopped)
}

type interceptMessage_cockroach1462 int

type localInterceptableTransport_cockroach1462 struct {
	mu      sync.Mutex
	Events  chan interceptMessage_cockroach1462
	stopper *Stopper_cockroach1462
}

func (lt *localInterceptableTransport_cockroach1462) Close() {}

type Transport_cockroach1462 interface {
	Close()
}

func NewLocalInterceptableTransport_cockroach1462(stopper *Stopper_cockroach1462) Transport_cockroach1462 {
	lt := &localInterceptableTransport_cockroach1462{
		Events:  make(chan interceptMessage_cockroach1462),
		stopper: stopper,
	}
	lt.start()
	return lt
}

func (lt *localInterceptableTransport_cockroach1462) start() {
	lt.stopper.RunWorker(func() {
		for {
			select {
			case <-lt.stopper.ShouldStop():
				return
			default:
				lt.Events <- interceptMessage_cockroach1462(0)
			}
		}
	})
}

func processEventsUntil_cockroach1462(ch <-chan interceptMessage_cockroach1462, stopper *Stopper_cockroach1462) {
	for {
		select {
		case _, ok := <-ch:
			runtime.Gosched()
			if !ok {
				return
			}
		case <-stopper.ShouldStop():
			return
		}
	}
}

func Cockroach1462() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(2000 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()
	for i := 0; i <= 1000; i++ {
		go func() { // G1
			stopper := NewStopper_cockroach1462()
			transport := NewLocalInterceptableTransport_cockroach1462(stopper).(*localInterceptableTransport_cockroach1462)
			stopper.RunWorker(func() {
				processEventsUntil_cockroach1462(transport.Events, stopper)
			})
			stopper.Stop()
		}()
	}
}

