// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"sync"
	"testing"
	"time"
)

func TestDebouncer(t *testing.T) {
	t.Parallel()
	type event struct {
		key       string
		order     uint64
		fired     bool
		wantFired bool
	}
	tests := []struct {
		label  string
		events []*event
	}{
		{
			label: "overridden",
			events: []*event{
				{key: "a", order: 1, wantFired: false},
				{key: "a", order: 2, wantFired: true},
			},
		},
		{
			label: "distinct labels",
			events: []*event{
				{key: "a", order: 1, wantFired: true},
				{key: "b", order: 2, wantFired: true},
			},
		},
		{
			label: "reverse order",
			events: []*event{
				{key: "a", order: 2, wantFired: true},
				{key: "a", order: 1, wantFired: false},
			},
		},
		{
			label: "multiple overrides",
			events: []*event{
				{key: "a", order: 1, wantFired: false},
				{key: "a", order: 2, wantFired: false},
				{key: "a", order: 3, wantFired: false},
				{key: "a", order: 4, wantFired: false},
				{key: "a", order: 5, wantFired: true},
			},
		},
	}
	for _, test := range tests {
		test := test
		t.Run(test.label, func(t *testing.T) {
			t.Parallel()
			d := newDebouncer()
			var wg sync.WaitGroup
			for i, e := range test.events {
				wg.Add(1)
				go func(e *event) {
					d.debounce(e.key, e.order, 500*time.Millisecond, func() {
						e.fired = true
					})
					wg.Done()
				}(e)
				// For a bit more fidelity, sleep to try to make things actually
				// execute in order. This doesn't have to be perfect, but could be done
				// properly using fake timers.
				if i < len(test.events)-1 {
					time.Sleep(10 * time.Millisecond)
				}
			}
			wg.Wait()
			for _, event := range test.events {
				if event.fired != event.wantFired {
					t.Errorf("(key: %q, order: %d): fired = %t, want %t",
						event.key, event.order, event.fired, event.wantFired)
				}
			}
		})
	}
}
