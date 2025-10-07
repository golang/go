// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

package main

import (
	"context"
	"os"
	"runtime/pprof"

	"sync"
	"time"
)

func init() {
	register("Istio18454", Istio18454)
}

const eventChCap_istio18454 = 1024

type Worker_istio18454 struct {
	ctx       context.Context
	ctxCancel context.CancelFunc
}

func (w *Worker_istio18454) Start(setupFn func(), runFn func(c context.Context)) {
	if setupFn != nil {
		setupFn()
	}
	go func() {
		runFn(w.ctx)
	}()
}

func (w *Worker_istio18454) Stop() {
	w.ctxCancel()
}

type Strategy_istio18454 struct {
	timer          *time.Timer
	timerFrequency time.Duration
	stateLock      sync.Mutex
	resetChan      chan struct{}
	worker         *Worker_istio18454
	startTimerFn   func()
}

func (s *Strategy_istio18454) OnChange() {
	s.stateLock.Lock()
	if s.timer != nil {
		s.stateLock.Unlock()
		s.resetChan <- struct{}{}
		return
	}
	s.startTimerFn()
	s.stateLock.Unlock()
}

func (s *Strategy_istio18454) startTimer() {
	s.timer = time.NewTimer(s.timerFrequency)
	eventLoop := func(ctx context.Context) {
		for {
			select {
			case <-s.timer.C:
			case <-s.resetChan:
				if !s.timer.Stop() {
					<-s.timer.C
				}
				s.timer.Reset(s.timerFrequency)
			case <-ctx.Done():
				s.timer.Stop()
				return
			}
		}
	}
	s.worker.Start(nil, eventLoop)
}

func (s *Strategy_istio18454) Close() {
	s.worker.Stop()
}

type Event_istio18454 int

type Processor_istio18454 struct {
	stateStrategy *Strategy_istio18454
	worker        *Worker_istio18454
	eventCh       chan Event_istio18454
}

func (p *Processor_istio18454) processEvent() {
	p.stateStrategy.OnChange()
}

func (p *Processor_istio18454) Start() {
	setupFn := func() {
		for i := 0; i < eventChCap_istio18454; i++ {
			p.eventCh <- Event_istio18454(0)
		}
	}
	runFn := func(ctx context.Context) {
		defer func() {
			p.stateStrategy.Close()
		}()
		for {
			select {
			case <-ctx.Done():
				return
			case <-p.eventCh:
				p.processEvent()
			}
		}
	}
	p.worker.Start(setupFn, runFn)
}

func (p *Processor_istio18454) Stop() {
	p.worker.Stop()
}

func NewWorker_istio18454() *Worker_istio18454 {
	worker := &Worker_istio18454{}
	worker.ctx, worker.ctxCancel = context.WithCancel(context.Background())
	return worker
}

func Istio18454() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()

	for i := 0; i < 100; i++ {
		go func() {
			stateStrategy := &Strategy_istio18454{
				timerFrequency: time.Nanosecond,
				resetChan:      make(chan struct{}, 1),
				worker:         NewWorker_istio18454(),
			}
			stateStrategy.startTimerFn = stateStrategy.startTimer

			p := &Processor_istio18454{
				stateStrategy: stateStrategy,
				worker:        NewWorker_istio18454(),
				eventCh:       make(chan Event_istio18454, eventChCap_istio18454),
			}

			p.Start()
			defer p.Stop()
		}()
	}
}
