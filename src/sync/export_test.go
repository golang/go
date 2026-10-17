// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sync

import "sync/atomic"

// Export for testing.
var Runtime_Semacquire = runtime_Semacquire
var Runtime_Semrelease = runtime_Semrelease
var Runtime_procPin = runtime_procPin
var Runtime_procUnpin = runtime_procUnpin

// State returns the state word and the semaphore count of wg.
func (wg *WaitGroup) State() (state uint64, sema uint32) {
	return wg.state.Load(), atomic.LoadUint32(&wg.sema)
}

// SetState sets the state word of wg.
func (wg *WaitGroup) SetState(state uint64) {
	wg.state.Store(state)
}

// PoolDequeue exports an interface for pollDequeue testing.
type PoolDequeue interface {
	PushHead(val any) bool
	PopHead() (any, bool)
	PopTail() (any, bool)
}

func NewPoolDequeue(n int) PoolDequeue {
	d := &poolDequeue{
		vals: make([]eface, n),
	}
	// For testing purposes, set the head and tail indexes close
	// to wrapping around.
	d.headTail.Store(d.pack(1<<dequeueBits-500, 1<<dequeueBits-500))
	return d
}

func (d *poolDequeue) PushHead(val any) bool {
	return d.pushHead(val)
}

func (d *poolDequeue) PopHead() (any, bool) {
	return d.popHead()
}

func (d *poolDequeue) PopTail() (any, bool) {
	return d.popTail()
}

func NewPoolChain() PoolDequeue {
	return new(poolChain)
}

func (c *poolChain) PushHead(val any) bool {
	c.pushHead(val)
	return true
}

func (c *poolChain) PopHead() (any, bool) {
	return c.popHead()
}

func (c *poolChain) PopTail() (any, bool) {
	return c.popTail()
}
