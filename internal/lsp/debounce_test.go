// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	errors "golang.org/x/xerrors"
)

func TestDebouncer(t *testing.T) {
	t.Parallel()

	const evtWait = 30 * time.Millisecond
	const initialDelay = 100 * time.Millisecond

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

			try := func(delay time.Duration) (bool, error) {
				d := newDebouncer()
				var wg sync.WaitGroup
				valid := true
				for i, e := range test.events {
					wg.Add(1)
					start := time.Now()
					go func(e *event) {
						if time.Since(start) > evtWait {
							// Due to slow scheduling this event is likely to have fired out
							// of order, so mark this attempt as invalid.
							valid = false
						}
						d.debounce(e.key, e.order, delay, func() {
							e.fired = true
						})
						wg.Done()
					}(e)
					// For a bit more fidelity, sleep to try to make things actually
					// execute in order. This shouldn't have to be perfect: as long as we
					// don't have extreme pauses the test should still pass.
					if i < len(test.events)-1 {
						time.Sleep(evtWait)
					}
				}
				wg.Wait()
				var errs []string
				for _, event := range test.events {
					if event.fired != event.wantFired {
						msg := fmt.Sprintf("(key: %q, order: %d): fired = %t, want %t",
							event.key, event.order, event.fired, event.wantFired)
						errs = append(errs, msg)
					}
				}
				var err error
				if len(errs) > 0 {
					err = errors.New(strings.Join(errs, "\n"))
				}
				// If the test took less than maxwait, no event before the
				return valid, err
			}

			if err := retryInvalid(100*time.Millisecond, try); err != nil {
				t.Error(err)
			}
		})
	}
}

// retryInvalid runs the try func up to three times with exponential back-off
// in its duration argument, starting with the provided initial duration. Calls
// to try are retried if their first result indicates that they may be invalid,
// and their second result is a non-nil error.
func retryInvalid(initial time.Duration, try func(time.Duration) (valid bool, err error)) (err error) {
	delay := initial
	for attempts := 3; attempts > 0; attempts-- {
		var valid bool
		valid, err = try(delay)
		if err == nil || valid {
			return err
		}
		delay += delay
	}
	return err
}
