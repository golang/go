// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Transport code's client connection pooling.

package http2

import (
	"net"
	"slices"
	"sync"
)

// clientConnPool is the HTTP/2 client connection pool.
// The pool never dials its own connections. Dials are done in net/http, and
// handed to this pool.
//
// TODO: use singleflight for addConnCalls?
type clientConnPool struct {
	mu sync.Mutex // TODO: maybe switch to RWMutex
	// TODO: add support for sharing conns based on cert names
	// (e.g. share conn for googleapis.com and appspot.com)
	conns        map[string][]*ClientConn // key is host:port
	keys         map[*ClientConn][]string
	addConnCalls map[string]*addConnCall // in-flight addConnIfNeeded calls
}

// GetClientConn returns a cached connection to addr,
// or ErrNoCachedConn if there is none available.
func (p *clientConnPool) GetClientConn(req *ClientRequest, addr string) (*ClientConn, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, cc := range p.conns[addr] {
		if cc.ReserveNewRequest() {
			// When a connection is presented to us by the net/http package,
			// the GetConn hook has already been called.
			// Don't call it a second time here.
			if !cc.getConnCalled {
				traceGetConn(req, addr)
			}
			cc.getConnCalled = false
			return cc, nil
		}
	}
	return nil, ErrNoCachedConn
}

// addConnIfNeeded makes a NewClientConn out of c if a connection for key doesn't
// already exist. It coalesces concurrent calls with the same key.
// This is used by the http1 Transport code when it creates a new connection. Because
// the http1 Transport doesn't de-dup TCP dials to outbound hosts (because it doesn't know
// the protocol), it can get into a situation where it has multiple TLS connections.
// This code decides which ones live or die.
// The return value used is whether c was used.
// c is never closed.
func (p *clientConnPool) addConnIfNeeded(key string, t *Transport, c net.Conn) (used bool, err error) {
	p.mu.Lock()
	for _, cc := range p.conns[key] {
		if cc.CanTakeNewRequest() {
			p.mu.Unlock()
			return false, nil
		}
	}
	call, dup := p.addConnCalls[key]
	if !dup {
		if p.addConnCalls == nil {
			p.addConnCalls = make(map[string]*addConnCall)
		}
		call = &addConnCall{
			p:    p,
			done: make(chan struct{}),
		}
		p.addConnCalls[key] = call
		go call.run(t, key, c)
	}
	p.mu.Unlock()

	<-call.done
	if call.err != nil {
		return false, call.err
	}
	return !dup, nil
}

type addConnCall struct {
	_    incomparable
	p    *clientConnPool
	done chan struct{} // closed when done
	err  error
}

func (c *addConnCall) run(t *Transport, key string, nc net.Conn) {
	cc, err := t.newClientConn(nc, t.disableKeepAlives(), nil)

	p := c.p
	p.mu.Lock()
	if err != nil {
		c.err = err
	} else {
		cc.getConnCalled = true // already called by the net/http package
		p.addConnLocked(key, cc)
	}
	delete(p.addConnCalls, key)
	p.mu.Unlock()
	close(c.done)
}

// p.mu must be held
func (p *clientConnPool) addConnLocked(key string, cc *ClientConn) {
	if slices.Contains(p.conns[key], cc) {
		return
	}
	if p.conns == nil {
		p.conns = make(map[string][]*ClientConn)
	}
	if p.keys == nil {
		p.keys = make(map[*ClientConn][]string)
	}
	p.conns[key] = append(p.conns[key], cc)
	p.keys[cc] = append(p.keys[cc], key)
}

func (p *clientConnPool) MarkDead(cc *ClientConn) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, key := range p.keys[cc] {
		vv, ok := p.conns[key]
		if !ok {
			continue
		}
		newList := filterOutClientConn(vv, cc)
		if len(newList) > 0 {
			p.conns[key] = newList
		} else {
			delete(p.conns, key)
		}
	}
	delete(p.keys, cc)
}

func (p *clientConnPool) closeIdleConnections() {
	p.mu.Lock()
	defer p.mu.Unlock()
	// TODO: don't close a cc if it was just added to the pool
	// milliseconds ago and has never been used. There's currently
	// a small race window with the HTTP/1 Transport's integration
	// where it can add an idle conn just before using it, and
	// somebody else can concurrently call CloseIdleConns and
	// break some caller's RoundTrip.
	for _, vv := range p.conns {
		for _, cc := range vv {
			cc.closeIfIdle()
		}
	}
}

func filterOutClientConn(in []*ClientConn, exclude *ClientConn) []*ClientConn {
	out := in[:0]
	for _, v := range in {
		if v != exclude {
			out = append(out, v)
		}
	}
	// If we filtered it out, zero out the last item to prevent
	// the GC from seeing it.
	if len(in) != len(out) {
		in[len(in)-1] = nil
	}
	return out
}
