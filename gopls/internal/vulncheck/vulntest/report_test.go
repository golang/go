// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

package vulntest

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func readAll(t *testing.T, filename string) io.Reader {
	d, err := os.ReadFile(filename)
	if err != nil {
		t.Fatal(err)
	}
	return bytes.NewReader(d)
}

func TestRoundTrip(t *testing.T) {
	// A report shouldn't change after being read and then written.
	in := filepath.Join("testdata", "report.yaml")
	r, err := readReport(readAll(t, in))
	if err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(t.TempDir(), "report.yaml")
	if err := r.Write(out); err != nil {
		t.Fatal(err)
	}

	want, err := os.ReadFile(in)
	if err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("mismatch (-want, +got):\n%s", diff)
	}
}
