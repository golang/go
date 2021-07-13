// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"testing"
	"time"
)

func TestDebouncer(t *testing.T) {
	t.Parallel()

	type event struct {
		key       string
		order     uint64
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
			d := newDebouncer()

			delays := make([]chan time.Time, len(test.events))
			okcs := make([]<-chan bool, len(test.events))

			// Register the events in deterministic order, synchronously.
			for i, e := range test.events {
				delays[i] = make(chan time.Time, 1)
				okcs[i] = d.debounce(e.key, e.order, delays[i])
			}

			// Now see which event fired.
			for i, okc := range okcs {
				event := test.events[i]
				delays[i] <- time.Now()
				fired := <-okc
				if fired != event.wantFired {
					t.Errorf("[key: %q, order: %d]: fired = %t, want %t", event.key, event.order, fired, event.wantFired)
				}
			}
		})
	}
}
