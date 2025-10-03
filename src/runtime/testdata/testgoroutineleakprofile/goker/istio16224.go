// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

package main

import (
	"os"
	"runtime/pprof"
	"sync"
	"time"
)

func init() {
	register("Istio16224", Istio16224)
}

type ConfigStoreCache_istio16224 interface {
	RegisterEventHandler(handler func())
	Run()
}

type Event_istio16224 int

type Handler_istio16224 func(Event_istio16224)

type configstoreMonitor_istio16224 struct {
	handlers []Handler_istio16224
	eventCh  chan Event_istio16224
}

func (m *configstoreMonitor_istio16224) Run(stop <-chan struct{}) {
	for {
		select {
		case <-stop:
			// This bug is not descibed, but is a true positive (in our eyes)
			// In a real run main exits when the goro is blocked here.
			if _, ok := <-m.eventCh; ok {
				close(m.eventCh)
			}
			return
		case ce, ok := <-m.eventCh:
			if ok {
				m.processConfigEvent(ce)
			}
		}
	}
}

func (m *configstoreMonitor_istio16224) processConfigEvent(ce Event_istio16224) {
	m.applyHandlers(ce)
}

func (m *configstoreMonitor_istio16224) AppendEventHandler(h Handler_istio16224) {
	m.handlers = append(m.handlers, h)
}

func (m *configstoreMonitor_istio16224) applyHandlers(e Event_istio16224) {
	for _, f := range m.handlers {
		f(e)
	}
}
func (m *configstoreMonitor_istio16224) ScheduleProcessEvent(configEvent Event_istio16224) {
	m.eventCh <- configEvent
}

type Monitor_istio16224 interface {
	Run(<-chan struct{})
	AppendEventHandler(Handler_istio16224)
	ScheduleProcessEvent(Event_istio16224)
}

type controller_istio16224 struct {
	monitor Monitor_istio16224
}

func (c *controller_istio16224) RegisterEventHandler(f func(Event_istio16224)) {
	c.monitor.AppendEventHandler(f)
}

func (c *controller_istio16224) Run(stop <-chan struct{}) {
	c.monitor.Run(stop)
}

func (c *controller_istio16224) Create() {
	c.monitor.ScheduleProcessEvent(Event_istio16224(0))
}

func NewMonitor_istio16224() Monitor_istio16224 {
	return NewBufferedMonitor_istio16224()
}

func NewBufferedMonitor_istio16224() Monitor_istio16224 {
	return &configstoreMonitor_istio16224{
		eventCh: make(chan Event_istio16224),
	}
}

func Istio16224() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()

	for i := 0; i < 100; i++ {
		go func() {
			controller := &controller_istio16224{monitor: NewMonitor_istio16224()}
			done := make(chan bool)
			lock := sync.Mutex{}
			controller.RegisterEventHandler(func(event Event_istio16224) {
				lock.Lock()
				defer lock.Unlock()
				done <- true
			})

			stop := make(chan struct{})
			go controller.Run(stop)

			controller.Create()

			lock.Lock() // blocks
			lock.Unlock()
			<-done

			close(stop)
		}()
	}
}
