// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package oldtrace

import (
	"bytes"
	"internal/trace/version"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCorruptedInputs(t *testing.T) {
	// These inputs crashed parser previously.
	tests := []string{
		"gotrace\x00\x020",
		"gotrace\x00Q00\x020",
		"gotrace\x00T00\x020",
		"gotrace\x00\xc3\x0200",
		"go 1.5 trace\x00\x00\x00\x00\x020",
		"go 1.5 trace\x00\x00\x00\x00Q00\x020",
		"go 1.5 trace\x00\x00\x00\x00T00\x020",
		"go 1.5 trace\x00\x00\x00\x00\xc3\x0200",
	}
	for _, data := range tests {
		res, err := Parse(strings.NewReader(data), 5)
		if err == nil || res.Events.Len() != 0 || res.Stacks != nil {
			t.Fatalf("no error on input: %q", data)
		}
	}
}

func TestParseCanned(t *testing.T) {
	files, err := os.ReadDir("./testdata")
	if err != nil {
		t.Fatalf("failed to read ./testdata: %v", err)
	}
	for _, f := range files {
		info, err := f.Info()
		if err != nil {
			t.Fatal(err)
		}
		if testing.Short() && info.Size() > 10000 {
			continue
		}
		name := filepath.Join("./testdata", f.Name())
		data, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		r := bytes.NewReader(data)
		v, err := version.ReadHeader(r)
		if err != nil {
			t.Errorf("failed to parse good trace %s: %s", f.Name(), err)
		}
		trace, err := Parse(r, v)
		switch {
		case strings.HasSuffix(f.Name(), "_good"):
			if err != nil {
				t.Errorf("failed to parse good trace %v: %v", f.Name(), err)
			}
			checkTrace(t, int(v), trace)
		case strings.HasSuffix(f.Name(), "_unordered"):
			if err != ErrTimeOrder {
				t.Errorf("unordered trace is not detected %v: %v", f.Name(), err)
			}
		default:
			t.Errorf("unknown input file suffix: %v", f.Name())
		}
	}
}

// checkTrace walks over a good trace and makes a bunch of additional checks
// that may not cause the parser to outright fail.
func checkTrace(t *testing.T, ver int, res Trace) {
	for i := 0; i < res.Events.Len(); i++ {
		ev := res.Events.Ptr(i)
		if ver >= 21 {
			if ev.Type == EvSTWStart && res.Strings[ev.Args[0]] == "unknown" {
				t.Errorf("found unknown STW event; update stwReasonStrings?")
			}
		}
	}
}

func TestBuckets(t *testing.T) {
	var evs Events

	const N = eventsBucketSize*3 + 123
	for i := 0; i < N; i++ {
		evs.append(Event{Ts: Timestamp(i)})
	}

	if n := len(evs.buckets); n != 4 {
		t.Fatalf("got %d buckets, want %d", n, 4)
	}

	if n := evs.Len(); n != N {
		t.Fatalf("got %d events, want %d", n, N)
	}

	var n int
	evs.All()(func(ev *Event) bool {
		n++
		return true
	})
	if n != N {
		t.Fatalf("iterated over %d events, expected %d", n, N)
	}

	const consume = eventsBucketSize + 50
	for i := 0; i < consume; i++ {
		if _, ok := evs.Pop(); !ok {
			t.Fatalf("iteration failed after %d events", i)
		}
	}

	if evs.buckets[0] != nil {
		t.Fatalf("expected first bucket to have been dropped")
	}
	for i, b := range evs.buckets[1:] {
		if b == nil {
			t.Fatalf("expected bucket %d to be non-nil", i+1)
		}
	}

	if n := evs.Len(); n != N-consume {
		t.Fatalf("got %d remaining elements, expected %d", n, N-consume)
	}

	ev := evs.Ptr(0)
	if ev.Ts != consume {
		t.Fatalf("got event %d, expected %d", int(ev.Ts), consume)
	}

	for {
		_, ok := evs.Pop()
		if !ok {
			break
		}
	}

	for i, b := range evs.buckets {
		if b != nil {
			t.Fatalf("expected bucket %d to be nil", i)
		}
	}
}
