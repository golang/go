// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build cgo

package runtime_test

import (
	"bytes"
	"fmt"
	"internal/testenv"
	"internal/trace"
	tracev2 "internal/trace/v2"
	"io"
	"os"
	"runtime"
	"strings"
	"testing"
)

// TestTraceUnwindCGO verifies that trace events emitted in cgo callbacks
// produce the same stack traces and don't cause any crashes regardless of
// tracefpunwindoff being set to 0 or 1.
func TestTraceUnwindCGO(t *testing.T) {
	if *flagQuick {
		t.Skip("-quick")
	}
	testenv.MustHaveGoBuild(t)
	t.Parallel()

	exe, err := buildTestProg(t, "testprogcgo")
	if err != nil {
		t.Fatal(err)
	}

	wantLogs := []string{
		"goCalledFromC",
		"goCalledFromCThread",
	}
	logs := make(map[string]*trace.Event)
	for _, category := range wantLogs {
		logs[category] = nil
	}
	logsV2 := make(map[string]*tracev2.Event)
	for _, category := range wantLogs {
		logsV2[category] = nil
	}
	for _, tracefpunwindoff := range []int{1, 0} {
		env := fmt.Sprintf("GODEBUG=tracefpunwindoff=%d", tracefpunwindoff)
		got := runBuiltTestProg(t, exe, "Trace", env)
		prefix, tracePath, found := strings.Cut(got, ":")
		if !found || prefix != "trace path" {
			t.Fatalf("unexpected output:\n%s\n", got)
		}
		defer os.Remove(tracePath)

		traceData, err := os.ReadFile(tracePath)
		if err != nil {
			t.Fatalf("failed to read trace: %s", err)
		}
		for category := range logs {
			event := mustFindLogV2(t, bytes.NewReader(traceData), category)
			if wantEvent := logsV2[category]; wantEvent == nil {
				logsV2[category] = &event
			} else if got, want := dumpStackV2(&event), dumpStackV2(wantEvent); got != want {
				t.Errorf("%q: got stack:\n%s\nwant stack:\n%s\n", category, got, want)
			}
		}
	}
}

// mustFindLog returns the EvUserLog event with the given category in events. It
// fails if no event or multiple events match the category.
func mustFindLog(t *testing.T, events []*trace.Event, category string) *trace.Event {
	t.Helper()
	var candidates []*trace.Event
	for _, e := range events {
		if e.Type == trace.EvUserLog && len(e.SArgs) >= 1 && e.SArgs[0] == category {
			candidates = append(candidates, e)
		}
	}
	if len(candidates) == 0 {
		t.Errorf("could not find log with category: %q", category)
	} else if len(candidates) > 1 {
		t.Errorf("found more than one log with category: %q", category)
	}
	return candidates[0]
}

// dumpStack returns e.Stk as a string.
func dumpStack(e *trace.Event) string {
	var buf bytes.Buffer
	for _, f := range e.Stk {
		file := strings.TrimPrefix(f.File, runtime.GOROOT())
		fmt.Fprintf(&buf, "%s\n\t%s:%d\n", f.Fn, file, f.Line)
	}
	return buf.String()
}

// parseTrace parses the given trace or skips the test if the trace is broken
// due to known issues. Partially copied from runtime/trace/trace_test.go.
func parseTrace(t *testing.T, r io.Reader) []*trace.Event {
	res, err := trace.Parse(r, "")
	if err == trace.ErrTimeOrder {
		t.Skipf("skipping trace: %v", err)
	}
	if err != nil {
		t.Fatalf("failed to parse trace: %v", err)
	}
	return res.Events
}

func mustFindLogV2(t *testing.T, trace io.Reader, category string) tracev2.Event {
	r, err := tracev2.NewReader(trace)
	if err != nil {
		t.Fatalf("bad trace: %v", err)
	}
	var candidates []tracev2.Event
	for {
		ev, err := r.ReadEvent()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("failed to parse trace: %v", err)
		}
		if ev.Kind() == tracev2.EventLog && ev.Log().Category == category {
			candidates = append(candidates, ev)
		}
	}
	if len(candidates) == 0 {
		t.Fatalf("could not find log with category: %q", category)
	} else if len(candidates) > 1 {
		t.Fatalf("found more than one log with category: %q", category)
	}
	return candidates[0]
}

// dumpStack returns e.Stack() as a string.
func dumpStackV2(e *tracev2.Event) string {
	var buf bytes.Buffer
	e.Stack().Frames(func(f tracev2.StackFrame) bool {
		file := strings.TrimPrefix(f.File, runtime.GOROOT())
		fmt.Fprintf(&buf, "%s\n\t%s:%d\n", f.Func, file, f.Line)
		return true
	})
	return buf.String()
}
