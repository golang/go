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
			if wantEvent := logs[category]; wantEvent == nil {
				logs[category] = &event
			} else if got, want := dumpStackV2(&event), dumpStackV2(wantEvent); got != want {
				t.Errorf("%q: got stack:\n%s\nwant stack:\n%s\n", category, got, want)
			}
		}
	}
}

func mustFindLogV2(t *testing.T, trc io.Reader, category string) trace.Event {
	r, err := trace.NewReader(trc)
	if err != nil {
		t.Fatalf("bad trace: %v", err)
	}
	var candidates []trace.Event
	for {
		ev, err := r.ReadEvent()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("failed to parse trace: %v", err)
		}
		if ev.Kind() == trace.EventLog && ev.Log().Category == category {
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
func dumpStackV2(e *trace.Event) string {
	var buf bytes.Buffer
	e.Stack().Frames(func(f trace.StackFrame) bool {
		file := strings.TrimPrefix(f.File, runtime.GOROOT())
		fmt.Fprintf(&buf, "%s\n\t%s:%d\n", f.Func, file, f.Line)
		return true
	})
	return buf.String()
}
