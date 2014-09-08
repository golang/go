// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package textproto

import (
	"sync"
)

// A Pipeline manages a pipelined in-order request/response sequence.
//
// To use a Pipeline p to manage multiple clients on a connection,
// each client should run:
//
//	id := p.Next()	// take a number
//
//	p.StartRequest(id)	// wait for turn to send request
//	«send request»
//	p.EndRequest(id)	// notify Pipeline that request is sent
//
//	p.StartResponse(id)	// wait for turn to read response
//	«read response»
//	p.EndResponse(id)	// notify Pipeline that response is read
//
// A pipelined server can use the same calls to ensure that
// responses computed in parallel are written in the correct order.
type Pipeline struct {
	mu       sync.Mutex
	id       uint
	request  sequencer
	response sequencer
}

// Next returns the next id for a request/response pair.
func (p *Pipeline) Next() uint {
	p.mu.Lock()
	id := p.id
	p.id++
	p.mu.Unlock()
	return id
}

// StartRequest blocks until it is time to send (or, if this is a server, receive)
// the request with the given id.
func (p *Pipeline) StartRequest(id uint) {
	p.request.Start(id)
}

// EndRequest notifies p that the request with the given id has been sent
// (or, if this is a server, received).
func (p *Pipeline) EndRequest(id uint) {
	p.request.End(id)
}

// StartResponse blocks until it is time to receive (or, if this is a server, send)
// the request with the given id.
func (p *Pipeline) StartResponse(id uint) {
	p.response.Start(id)
}

// EndResponse notifies p that the response with the given id has been received
// (or, if this is a server, sent).
func (p *Pipeline) EndResponse(id uint) {
	p.response.End(id)
}

// A sequencer schedules a sequence of numbered events that must
// happen in order, one after the other.  The event numbering must start
// at 0 and increment without skipping.  The event number wraps around
// safely as long as there are not 2^32 simultaneous events pending.
type sequencer struct {
	mu   sync.Mutex
	id   uint
	wait map[uint]chan uint
}

// Start waits until it is time for the event numbered id to begin.
// That is, except for the first event, it waits until End(id-1) has
// been called.
func (s *sequencer) Start(id uint) {
	s.mu.Lock()
	if s.id == id {
		s.mu.Unlock()
		return
	}
	c := make(chan uint)
	if s.wait == nil {
		s.wait = make(map[uint]chan uint)
	}
	s.wait[id] = c
	s.mu.Unlock()
	<-c
}

// End notifies the sequencer that the event numbered id has completed,
// allowing it to schedule the event numbered id+1.  It is a run-time error
// to call End with an id that is not the number of the active event.
func (s *sequencer) End(id uint) {
	s.mu.Lock()
	if s.id != id {
		panic("out of sync")
	}
	id++
	s.id = id
	if s.wait == nil {
		s.wait = make(map[uint]chan uint)
	}
	c, ok := s.wait[id]
	if ok {
		delete(s.wait, id)
	}
	s.mu.Unlock()
	if ok {
		c <- 1
	}
}
