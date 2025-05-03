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
	"strings"
	"testing"

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
			testReader(t, tr, ver, exp)
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

func testReader(t *testing.T, tr io.Reader, ver version.Version, exp *testtrace.Expectation) {
	r, err := trace.NewReader(tr)
	if err != nil {
		if err := exp.Check(err); err != nil {
			t.Error(err)
		}
		return
	}
	v := testtrace.NewValidator()
	v.GoVersion = ver
	for {
		ev, err := r.ReadEvent()
		if err == io.EOF {
			break
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
