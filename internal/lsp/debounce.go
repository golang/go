// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"sync"
	"time"
)

type debounceFunc struct {
	order uint64
	done  chan struct{}
}

type debouncer struct {
	mu    sync.Mutex
	funcs map[string]*debounceFunc
}

func newDebouncer() *debouncer {
	return &debouncer{
		funcs: make(map[string]*debounceFunc),
	}
}

// debounce waits timeout before running f, if no subsequent call is made with
// the same key in the intervening time. If a later call to debounce with the
// same key occurs while the original call is blocking, the original call will
// return immediately without running its f.
//
// If order is specified, it will be used to order calls logically, so calls
// with lesser order will not cancel calls with greater order.
func (d *debouncer) debounce(key string, order uint64, timeout time.Duration, f func()) {
	if timeout == 0 {
		// Degenerate case: no debouncing.
		f()
		return
	}

	// First, atomically acquire the current func, cancel it, and insert this
	// call into d.funcs.
	d.mu.Lock()
	current, ok := d.funcs[key]
	if ok && current.order > order {
		// If we have a logical ordering of events (as is the case for snapshots),
		// don't overwrite a later event with an earlier event.
		d.mu.Unlock()
		return
	}
	if ok {
		close(current.done)
	}
	done := make(chan struct{})
	next := &debounceFunc{
		order: order,
		done:  done,
	}
	d.funcs[key] = next
	d.mu.Unlock()

	// Next, wait to be cancelled or for our wait to expire. There is a race here
	// that we must handle: our timer could expire while another goroutine holds
	// d.mu.
	select {
	case <-done:
	case <-time.After(timeout):
		d.mu.Lock()
		if d.funcs[key] != next {
			// We lost the race: another event has arrived for the key and started
			// waiting. We could reasonably choose to run f at this point, but doing
			// nothing is simpler.
			d.mu.Unlock()
			return
		}
		delete(d.funcs, key)
		d.mu.Unlock()
		f()
	}
}
