// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	. "runtime"
	"strconv"
	"sync"
	"testing"
)

func TestTraceMap(t *testing.T) {
	var m TraceMap

	// Try all these operations multiple times between resets, to make sure
	// we're resetting properly.
	for range 3 {
		var d = [...]string{
			"a",
			"b",
			"aa",
			"ab",
			"ba",
			"bb",
		}
		for i, s := range d {
			id, inserted := m.PutString(s)
			if !inserted {
				t.Errorf("expected to have inserted string %q, but did not", s)
			}
			if id != uint64(i+1) {
				t.Errorf("expected string %q to have ID %d, but got %d instead", s, i+1, id)
			}
		}
		for i, s := range d {
			id, inserted := m.PutString(s)
			if inserted {
				t.Errorf("inserted string %q, but expected to have not done so", s)
			}
			if id != uint64(i+1) {
				t.Errorf("expected string %q to have ID %d, but got %d instead", s, i+1, id)
			}
		}
		m.Reset()
	}
}

func TestTraceMapConcurrent(t *testing.T) {
	var m TraceMap

	var wg sync.WaitGroup
	for i := range 3 {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()

			si := strconv.Itoa(i)
			var d = [...]string{
				"a" + si,
				"b" + si,
				"aa" + si,
				"ab" + si,
				"ba" + si,
				"bb" + si,
			}
			ids := make([]uint64, 0, len(d))
			for _, s := range d {
				id, inserted := m.PutString(s)
				if !inserted {
					t.Errorf("expected to have inserted string %q, but did not", s)
				}
				ids = append(ids, id)
			}
			for i, s := range d {
				id, inserted := m.PutString(s)
				if inserted {
					t.Errorf("inserted string %q, but expected to have not done so", s)
				}
				if id != ids[i] {
					t.Errorf("expected string %q to have ID %d, but got %d instead", s, ids[i], id)
				}
			}
		}(i)
	}
	wg.Wait()
	m.Reset()
}
