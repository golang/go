// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"sync"
	"time"
)

type debounceEvent struct {
	order uint64
	done  chan struct{}
}

type debouncer struct {
	mu     sync.Mutex
	events map[string]*debounceEvent
}

func newDebouncer() *debouncer {
	return &debouncer{
		events: make(map[string]*debounceEvent),
	}
}

// debounce returns a channel that receives a boolean reporting whether,
// by the time the delay channel receives a value, this call is (or will be)
// the most recent call with the highest order number for its key.
func (d *debouncer) debounce(key string, order uint64, delay <-chan time.Time) <-chan bool {
	okc := make(chan bool, 1)

	d.mu.Lock()
	if prev, ok := d.events[key]; ok {
		if prev.order > order {
			// If we have a logical ordering of events (as is the case for snapshots),
			// don't overwrite a later event with an earlier event.
			d.mu.Unlock()
			okc <- false
			return okc
		}
		close(prev.done)
	}
	done := make(chan struct{})
	next := &debounceEvent{
		order: order,
		done:  done,
	}
	d.events[key] = next
	d.mu.Unlock()

	go func() {
		ok := false
		select {
		case <-delay:
			d.mu.Lock()
			if d.events[key] == next {
				ok = true
				delete(d.events, key)
			} else {
				// The event was superseded before we acquired d.mu.
			}
			d.mu.Unlock()
		case <-done:
		}
		okc <- ok
	}()

	return okc
}
