// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sync

// Once is an object that will perform exactly one action.
type Once struct {
	m    Mutex
	done bool
}

// Do calls the function f if and only if the method is being called for the
// first time with this receiver.  In other words, given
// 	var once Once
// if Do(f) is called multiple times, only the first call will invoke f,
// even if f has a different value in each invocation.  A new instance of
// Once is required for each function to execute.
//
// Do is intended for initialization that must be run exactly once.  Since f
// is niladic, it may be necessary to use a function literal to capture the
// arguments to a function to be invoked by Do:
// 	config.once.Do(func() { config.init(filename) })
//
// Because no call to Do returns until the one call to f returns, if f causes
// Do to be called, it will deadlock.
//
func (o *Once) Do(f func()) {
	o.m.Lock()
	defer o.m.Unlock()
	if !o.done {
		o.done = true
		f()
	}
}
