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

	"internal/trace/v2"
	"internal/trace/v2/raw"
	"internal/trace/v2/testtrace"
	"internal/trace/v2/version"
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
			tr, exp, err := testtrace.ParseFile(testPath)
			if err != nil {
				t.Fatalf("failed to parse test file at %s: %v", testPath, err)
			}
			testReader(t, tr, exp)
		})
	}
}

func testReader(t *testing.T, tr io.Reader, exp *testtrace.Expectation) {
	r, err := trace.NewReader(tr)
	if err != nil {
		if err := exp.Check(err); err != nil {
			t.Error(err)
		}
		return
	}
	v := testtrace.NewValidator()
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
	tw, err := raw.NewTextWriter(&sb, version.Go122)
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
