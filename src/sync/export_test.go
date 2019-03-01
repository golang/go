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
	return &poolDequeue{
		vals: make([]eface, n),
	}
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
