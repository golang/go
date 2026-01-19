// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

package main

import (
	"os"
	"runtime/pprof"
	"sync"
	"sync/atomic"
	"time"
)

func init() {
	register("Cockroach1055", Cockroach1055)
}

type Stopper_cockroach1055 struct {
	stopper  chan struct{}
	stop     sync.WaitGroup
	mu       sync.Mutex
	draining int32
	drain    sync.WaitGroup
}

func (s *Stopper_cockroach1055) AddWorker() {
	s.stop.Add(1)
}

func (s *Stopper_cockroach1055) ShouldStop() <-chan struct{} {
	if s == nil {
		return nil
	}
	return s.stopper
}

func (s *Stopper_cockroach1055) SetStopped() {
	if s != nil {
		s.stop.Done()
	}
}

func (s *Stopper_cockroach1055) Quiesce() {
	s.mu.Lock()
	defer s.mu.Unlock()
	atomic.StoreInt32(&s.draining, 1)
	s.drain.Wait()
	atomic.StoreInt32(&s.draining, 0)
}

func (s *Stopper_cockroach1055) Stop() {
	s.mu.Lock() // L1
	defer s.mu.Unlock()
	atomic.StoreInt32(&s.draining, 1)
	s.drain.Wait()
	close(s.stopper)
	s.stop.Wait()
}

func (s *Stopper_cockroach1055) StartTask() bool {
	if atomic.LoadInt32(&s.draining) == 0 {
		s.mu.Lock()
		defer s.mu.Unlock()
		s.drain.Add(1)
		return true
	}
	return false
}

func NewStopper_cockroach1055() *Stopper_cockroach1055 {
	return &Stopper_cockroach1055{
		stopper: make(chan struct{}),
	}
}

func Cockroach1055() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()

	for i := 0; i <= 1000; i++ {
		go func() { // G1
			var stoppers []*Stopper_cockroach1055
			for i := 0; i < 2; i++ {
				stoppers = append(stoppers, NewStopper_cockroach1055())
			}

			for i := range stoppers {
				s := stoppers[i]
				s.AddWorker()
				go func() { // G2
					s.StartTask()
					<-s.ShouldStop()
					s.SetStopped()
				}()
			}

			done := make(chan struct{})
			go func() { // G3
				for _, s := range stoppers {
					s.Quiesce()
				}
				for _, s := range stoppers {
					s.Stop()
				}
				close(done)
			}()

			<-done
		}()
	}
}
