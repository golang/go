// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

package main

import (
	"os"
	"runtime"
	"runtime/pprof"
	"time"
)

func init() {
	register("Cockroach2448", Cockroach2448)
}

type Stopper_cockroach2448 struct {
	Done chan bool
}

func (s *Stopper_cockroach2448) ShouldStop() <-chan bool {
	return s.Done
}

type EventMembershipChangeCommitted_cockroach2448 struct {
	Callback func()
}

type MultiRaft_cockroach2448 struct {
	stopper      *Stopper_cockroach2448
	Events       chan interface{}
	callbackChan chan func()
}

// sendEvent can be invoked many times
func (m *MultiRaft_cockroach2448) sendEvent(event interface{}) {
	select {
	case m.Events <- event: // Waiting for events consumption
	case <-m.stopper.ShouldStop():
	}
}

type state_cockroach2448 struct {
	*MultiRaft_cockroach2448
}

func (s *state_cockroach2448) start() {
	for {
		select {
		case <-s.stopper.ShouldStop():
			return
		case cb := <-s.callbackChan:
			cb()
		default:
			s.handleWriteResponse()
			time.Sleep(100 * time.Microsecond)
		}
	}
}

func (s *state_cockroach2448) handleWriteResponse() {
	s.sendEvent(&EventMembershipChangeCommitted_cockroach2448{
		Callback: func() {
			select {
			case s.callbackChan <- func() { // Waiting for callbackChan consumption
				time.Sleep(time.Nanosecond)
			}:
			case <-s.stopper.ShouldStop():
			}
		},
	})
}

type Store_cockroach2448 struct {
	multiraft *MultiRaft_cockroach2448
}

func (s *Store_cockroach2448) processRaft() {
	for {
		select {
		case e := <-s.multiraft.Events:
			switch e := e.(type) {
			case *EventMembershipChangeCommitted_cockroach2448:
				callback := e.Callback
				runtime.Gosched()
				if callback != nil {
					callback() // Waiting for callbackChan consumption
				}
			}
		case <-s.multiraft.stopper.ShouldStop():
			return
		}
	}
}

func NewStoreAndState_cockroach2448() (*Store_cockroach2448, *state_cockroach2448) {
	stopper := &Stopper_cockroach2448{
		Done: make(chan bool),
	}
	mltrft := &MultiRaft_cockroach2448{
		stopper:      stopper,
		Events:       make(chan interface{}),
		callbackChan: make(chan func()),
	}
	st := &state_cockroach2448{mltrft}
	s := &Store_cockroach2448{mltrft}
	return s, st
}

func Cockroach2448() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()
	for i := 0; i < 1000; i++ {
		go func() {
			s, st := NewStoreAndState_cockroach2448()
			go s.processRaft() // G1
			go st.start()      // G2
		}()
	}
}

