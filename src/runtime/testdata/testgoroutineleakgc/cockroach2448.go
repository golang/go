package main

import (
	"runtime"
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
			time.Sleep(time.Millisecond)
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
	defer func() {
		time.Sleep(time.Second)
		runtime.GC()
	}()
	for i := 0; i < 1000; i++ {
		go func() {
			s, st := NewStoreAndState_cockroach2448()
			// deadlocks: x > 0
			go s.processRaft() // G1
			// deadlocks: x > 0
			go st.start() // G2
		}()
	}
}

// Example of deadlock trace:
//
//	G1													G2
//	--------------------------------------------------------------------------------------------------
//	s.processRaft()										st.start()
//	select												.
//	.													select [default]
//	.													s.handleWriteResponse()
//	.													s.sendEvent()
//	.													select
//	<-s.multiraft.Events <---->							m.Events <- event
//	.													select [default]
//	.													s.handleWriteResponse()
//	.													s.sendEvent()
//	.													select [m.Events<-, <-s.stopper.ShouldStop()]
//	callback()
//	select [m.callbackChan<-,<-s.stopper.ShouldStop()]	.
