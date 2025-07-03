// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace_test

import (
	"bytes"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"internal/trace"
	"internal/trace/raw"
	"internal/trace/testtrace"
	"internal/trace/version"
)

var (
	logEvents  = flag.Bool("log-events", false, "whether to log high-level events; significantly slows down tests")
	dumpTraces = flag.Bool("dump-traces", false, "dump traces even on success")
)

func TestReaderGolden(t *testing.T) {
	matches, err := filepath.Glob("./testdata/tests/*.test")
	if err != nil {
		t.Fatalf("failed to glob for tests: %v", err)
	}
	for _, testPath := range matches {
		testPath := testPath
		testName, err := filepath.Rel("./testdata", testPath)
		if err != nil {
			t.Fatalf("failed to relativize testdata path: %v", err)
		}
		t.Run(testName, func(t *testing.T) {
			tr, ver, exp, err := testtrace.ParseFile(testPath)
			if err != nil {
				t.Fatalf("failed to parse test file at %s: %v", testPath, err)
			}
			v := testtrace.NewValidator()
			v.GoVersion = ver
			testReader(t, tr, v, exp)
		})
	}
}

func FuzzReader(f *testing.F) {
	// Currently disabled because the parser doesn't do much validation and most
	// getters can be made to panic. Turn this on once the parser is meant to
	// reject invalid traces.
	const testGetters = false

	f.Fuzz(func(t *testing.T, b []byte) {
		r, err := trace.NewReader(bytes.NewReader(b))
		if err != nil {
			return
		}
		for {
			ev, err := r.ReadEvent()
			if err != nil {
				break
			}

			if !testGetters {
				continue
			}
			// Make sure getters don't do anything that panics
			switch ev.Kind() {
			case trace.EventLabel:
				ev.Label()
			case trace.EventLog:
				ev.Log()
			case trace.EventMetric:
				ev.Metric()
			case trace.EventRangeActive, trace.EventRangeBegin:
				ev.Range()
			case trace.EventRangeEnd:
				ev.Range()
				ev.RangeAttributes()
			case trace.EventStateTransition:
				ev.StateTransition()
			case trace.EventRegionBegin, trace.EventRegionEnd:
				ev.Region()
			case trace.EventTaskBegin, trace.EventTaskEnd:
				ev.Task()
			case trace.EventSync:
			case trace.EventStackSample:
			case trace.EventBad:
			}
		}
	})
}

func testReader(t *testing.T, tr io.Reader, v *testtrace.Validator, exp *testtrace.Expectation) {
	r, err := trace.NewReader(tr)
	if err != nil {
		if err := exp.Check(err); err != nil {
			t.Error(err)
		}
		return
	}
	for {
		ev, err := r.ReadEvent()
		if err == io.EOF {
			break
		}
		v.GoVersion = r.GoVersion()
		if runtime.GOOS == "windows" || runtime.GOARCH == "wasm" {
			v.SkipClockSnapshotChecks()
		}
		if err != nil {
			if err := exp.Check(err); err != nil {
				t.Error(err)
			}
			return
		}
		if *logEvents {
			t.Log(ev.String())
		}
		if err := v.Event(ev); err != nil {
			t.Error(err)
		}
	}
	if err := exp.Check(nil); err != nil {
		t.Error(err)
	}
}

func dumpTraceToText(t *testing.T, b []byte) string {
	t.Helper()

	br, err := raw.NewReader(bytes.NewReader(b))
	if err != nil {
		t.Fatalf("dumping trace: %v", err)
	}
	var sb strings.Builder
	tw, err := raw.NewTextWriter(&sb, version.Current)
	if err != nil {
		t.Fatalf("dumping trace: %v", err)
	}
	for {
		ev, err := br.ReadEvent()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("dumping trace: %v", err)
		}
		if err := tw.WriteEvent(ev); err != nil {
			t.Fatalf("dumping trace: %v", err)
		}
	}
	return sb.String()
}

func dumpTraceToFile(t *testing.T, testName string, stress bool, b []byte) string {
	t.Helper()

	desc := "default"
	if stress {
		desc = "stress"
	}
	name := fmt.Sprintf("%s.%s.trace.", testName, desc)
	f, err := os.CreateTemp("", name)
	if err != nil {
		t.Fatalf("creating temp file: %v", err)
	}
	defer f.Close()
	if _, err := io.Copy(f, bytes.NewReader(b)); err != nil {
		t.Fatalf("writing trace dump to %q: %v", f.Name(), err)
	}
	return f.Name()
}

func TestTraceGenSync(t *testing.T) {
	type sync struct {
		Time          trace.Time
		ClockSnapshot *trace.ClockSnapshot
	}
	runTest := func(testName string, wantSyncs []sync) {
		t.Run(testName, func(t *testing.T) {
			testPath := "testdata/tests/" + testName
			r, _, _, err := testtrace.ParseFile(testPath)
			if err != nil {
				t.Fatalf("malformed test %s: bad trace file: %v", testPath, err)
			}
			tr, err := trace.NewReader(r)
			if err != nil {
				t.Fatalf("malformed test %s: bad trace file: %v", testPath, err)
			}
			var syncEvents []trace.Event
			for {
				ev, err := tr.ReadEvent()
				if err == io.EOF {
					break
				}
				if err != nil {
					t.Fatalf("malformed test %s: bad trace file: %v", testPath, err)
				}
				if ev.Kind() == trace.EventSync {
					syncEvents = append(syncEvents, ev)
				}
			}

			if got, want := len(syncEvents), len(wantSyncs); got != want {
				t.Errorf("got %d sync events, want %d", got, want)
			}

			for i, want := range wantSyncs {
				got := syncEvents[i]
				gotSync := syncEvents[i].Sync()
				if got.Time() != want.Time {
					t.Errorf("sync=%d got time %d, want %d", i+1, got.Time(), want.Time)
				}
				if gotSync.ClockSnapshot == nil && want.ClockSnapshot == nil {
					continue
				}
				if gotSync.ClockSnapshot.Trace != want.ClockSnapshot.Trace {
					t.Errorf("sync=%d got trace time %d, want %d", i+1, gotSync.ClockSnapshot.Trace, want.ClockSnapshot.Trace)
				}
				if !gotSync.ClockSnapshot.Wall.Equal(want.ClockSnapshot.Wall) {
					t.Errorf("sync=%d got wall time %s, want %s", i+1, gotSync.ClockSnapshot.Wall, want.ClockSnapshot.Wall)
				}
				if gotSync.ClockSnapshot.Mono != want.ClockSnapshot.Mono {
					t.Errorf("sync=%d got mono time %d, want %d", i+1, gotSync.ClockSnapshot.Mono, want.ClockSnapshot.Mono)
				}
			}
		})
	}

	runTest("go123-sync.test", []sync{
		{10, nil},
		{40, nil},
		// The EvFrequency batch for generation 3 is emitted at trace.Time(80),
		// but 60 is the minTs of the generation, see b30 in the go generator.
		{60, nil},
		{63, nil},
	})

	runTest("go125-sync.test", []sync{
		{9, &trace.ClockSnapshot{Trace: 10, Mono: 99, Wall: time.Date(2025, 2, 28, 15, 4, 9, 123, time.UTC)}},
		{38, &trace.ClockSnapshot{Trace: 40, Mono: 199, Wall: time.Date(2025, 2, 28, 15, 4, 10, 123, time.UTC)}},
		{58, &trace.ClockSnapshot{Trace: 60, Mono: 299, Wall: time.Date(2025, 2, 28, 15, 4, 11, 123, time.UTC)}},
		{83, nil},
	})
}
