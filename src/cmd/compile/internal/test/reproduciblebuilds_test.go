// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"bytes"
	"internal/testenv"
	"os"
	"path/filepath"
	"testing"
)

func TestReproducibleBuilds(t *testing.T) {
	tests := []struct {
		file string
		// Number of concurrent backend compilations. More of them exposes
		// more of the nondeterminism that comes from the runtime scheduler,
		// but costs more to run.
		conc string
	}{
		{"issue20272.go", "2"},
		{"issue27013.go", "2"},
		{"issue30202.go", "2"},
		{"issue82043.go", "16"},
	}

	testenv.MustHaveGoBuild(t)
	iters := 10
	if testing.Short() {
		iters = 4
	}
	t.Parallel()
	for _, test := range tests {
		t.Run(test.file, func(t *testing.T) {
			t.Parallel()
			var want []byte
			tmp, err := os.CreateTemp("", "")
			if err != nil {
				t.Fatalf("temp file creation failed: %v", err)
			}
			defer os.Remove(tmp.Name())
			defer tmp.Close()
			for i := 0; i < iters; i++ {
				out, err := testenv.Command(t, testenv.GoToolPath(t), "tool", "compile", "-p=p", "-c", test.conc, "-o", tmp.Name(), filepath.Join("testdata", "reproducible", test.file)).CombinedOutput()
				if err != nil {
					t.Fatalf("failed to compile: %v\n%s", err, out)
				}
				obj, err := os.ReadFile(tmp.Name())
				if err != nil {
					t.Fatalf("failed to read object file: %v", err)
				}
				if i == 0 {
					want = obj
				} else {
					if !bytes.Equal(want, obj) {
						t.Fatalf("builds produced different output after %d iters (%d bytes vs %d bytes)", i, len(want), len(obj))
					}
				}
			}
		})
	}
}

func TestIssue38068(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	t.Parallel()

	// Compile a small package with and without the concurrent
	// backend, then check to make sure that the resulting archives
	// are identical.  Note: this uses "go tool compile" instead of
	// "go build" since the latter will generate different build IDs
	// if it sees different command line flags.
	scenarios := []struct {
		tag     string
		args    string
		libpath string
	}{
		{tag: "serial", args: "-c=1"},
		{tag: "concurrent", args: "-c=2"}}

	tmpdir := t.TempDir()

	src := filepath.Join("testdata", "reproducible", "issue38068.go")
	for i := range scenarios {
		s := &scenarios[i]
		s.libpath = filepath.Join(tmpdir, s.tag+".a")
		// Note: use of "-p" required in order for DWARF to be generated.
		cmd := testenv.Command(t, testenv.GoToolPath(t), "tool", "compile", "-p=issue38068", "-buildid=", s.args, "-o", s.libpath, src)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("%v: %v:\n%s", cmd.Args, err, out)
		}
	}

	readBytes := func(fn string) []byte {
		payload, err := os.ReadFile(fn)
		if err != nil {
			t.Fatalf("failed to read executable '%s': %v", fn, err)
		}
		return payload
	}

	b1 := readBytes(scenarios[0].libpath)
	b2 := readBytes(scenarios[1].libpath)
	if !bytes.Equal(b1, b2) {
		t.Fatalf("concurrent and serial builds produced different output")
	}
}
