// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sync

// Export for testing.
var Runtime_Semacquire = runtime_Semacquire
var Runtime_Semrelease = runtime_Semrelease
var Runtime_procPin = runtime_procPin
var Runtime_procUnpin = runtime_procUnpin

// poolDequeue testing.
type PoolDequeue interface {
	PushHead(val interface{}) bool
	PopHead() (interface{}, bool)
	PopTail() (interface{}, bool)
}

func NewPoolDequeue(n int) PoolDequeue {
	d := &poolDequeue{
		vals: make([]eface, n),
	}
	// For testing purposes, set the head and tail indexes close
	// to wrapping around.
	d.headTail = d.pack(1<<dequeueBits-500, 1<<dequeueBits-500)
	return d
}

func (d *poolDequeue) PushHead(val interface{}) bool {
	return d.pushHead(val)
}

func (d *poolDequeue) PopHead() (interface{}, bool) {
	return d.popHead()
}

func (d *poolDequeue) PopTail() (interface{}, bool) {
	return d.popTail()
}

func NewPoolChain() PoolDequeue {
	return new(poolChain)
}

func (c *poolChain) PushHead(val interface{}) bool {
	c.pushHead(val)
	return true
}

func (c *poolChain) PopHead() (interface{}, bool) {
	return c.popHead()
}

func (c *poolChain) PopTail() (interface{}, bool) {
	return c.popTail()
}
