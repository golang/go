// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

import (
	"bytes"
	"io/ioutil"
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
		res, err := Parse(strings.NewReader(data), "")
		if err == nil || res.Events != nil || res.Stacks != nil {
			t.Fatalf("no error on input: %q", data)
		}
	}
}

func TestParseCanned(t *testing.T) {
	files, err := ioutil.ReadDir("./testdata")
	if err != nil {
		t.Fatalf("failed to read ./testdata: %v", err)
	}
	for _, f := range files {
		data, err := ioutil.ReadFile(filepath.Join("./testdata", f.Name()))
		if err != nil {
			t.Fatalf("failed to read input file: %v", err)
		}
		// Instead of Parse that requires a proper binary name for old traces,
		// we use 'parse' that omits symbol lookup if an empty string is given.
		_, _, err = parse(bytes.NewReader(data), "")
		switch {
		case strings.HasSuffix(f.Name(), "_good"):
			if err != nil {
				t.Errorf("failed to parse good trace %v: %v", f.Name(), err)
			}
		case strings.HasSuffix(f.Name(), "_unordered"):
			if err != ErrTimeOrder {
				t.Errorf("unordered trace is not detected %v: %v", f.Name(), err)
			}
		default:
			t.Errorf("unknown input file suffix: %v", f.Name())
		}
	}
}

func TestParseVersion(t *testing.T) {
	tests := map[string]int{
		"go 1.5 trace\x00\x00\x00\x00": 1005,
		"go 1.7 trace\x00\x00\x00\x00": 1007,
		"go 1.10 trace\x00\x00\x00":    1010,
		"go 1.25 trace\x00\x00\x00":    1025,
		"go 1.234 trace\x00\x00":       1234,
		"go 1.2345 trace\x00":          -1,
		"go 0.0 trace\x00\x00\x00\x00": -1,
		"go a.b trace\x00\x00\x00\x00": -1,
	}
	for header, ver := range tests {
		ver1, err := parseHeader([]byte(header))
		if ver == -1 {
			if err == nil {
				t.Fatalf("no error on input: %q, version %v", header, ver1)
			}
		} else {
			if err != nil {
				t.Fatalf("failed to parse: %q (%v)", header, err)
			}
			if ver != ver1 {
				t.Fatalf("wrong version: %v, want %v, input: %q", ver1, ver, header)
			}
		}
	}
}

func TestTimestampOverflow(t *testing.T) {
	// Test that parser correctly handles large timestamps (long tracing).
	w := NewWriter()
	w.Emit(EvBatch, 0, 0)
	w.Emit(EvFrequency, 1e9)
	for ts := uint64(1); ts < 1e16; ts *= 2 {
		w.Emit(EvGoCreate, ts, ts, 0, 0)
	}
	if _, err := Parse(w, ""); err != nil {
		t.Fatalf("failed to parse: %v", err)
	}
}
