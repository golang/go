// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

var (
	updateGolden = flag.Bool("updategolden", false, "update golden files")
)

func TestGolden(t *testing.T) {
	td, err := os.Open("testdata/src/pkg")
	if err != nil {
		t.Fatal(err)
	}
	fis, err := td.Readdir(0)
	if err != nil {
		t.Fatal(err)
	}
	for _, fi := range fis {
		if !fi.IsDir() {
			continue
		}
		w := NewWalker()
		w.wantedPkg[fi.Name()] = true

		w.root = "testdata/src/pkg"
		goldenFile := filepath.Join("testdata", "src", "pkg", fi.Name(), "golden.txt")
		w.WalkPackage(fi.Name())

		if *updateGolden {
			os.Remove(goldenFile)
			f, err := os.Create(goldenFile)
			if err != nil {
				t.Fatal(err)
			}
			for _, feat := range w.Features() {
				fmt.Fprintf(f, "%s\n", feat)
			}
			f.Close()
		}

		bs, err := ioutil.ReadFile(goldenFile)
		if err != nil {
			t.Fatalf("opening golden.txt for package %q: %v", fi.Name(), err)
		}
		wanted := strings.Split(string(bs), "\n")
		sort.Strings(wanted)
		for _, feature := range wanted {
			if feature == "" {
				continue
			}
			_, ok := w.features[feature]
			if !ok {
				t.Errorf("package %s: missing feature %q", fi.Name(), feature)
			}
			delete(w.features, feature)
		}

		for _, feature := range w.Features() {
			t.Errorf("package %s: extra feature not in golden file: %q", fi.Name(), feature)
		}
	}
}

func TestCompareAPI(t *testing.T) {
	tests := []struct {
		name                                    string
		features, required, optional, exception []string
		ok                                      bool   // want
		out                                     string // want
	}{
		{
			name:     "feature added",
			features: []string{"A", "B", "C", "D", "E", "F"},
			required: []string{"B", "D"},
			ok:       true,
			out:      "+A\n+C\n+E\n+F\n",
		},
		{
			name:     "feature removed",
			features: []string{"C", "A"},
			required: []string{"A", "B", "C"},
			ok:       false,
			out:      "-B\n",
		},
		{
			name:     "feature added then removed",
			features: []string{"A", "C"},
			optional: []string{"B"},
			required: []string{"A", "C"},
			ok:       true,
			out:      "±B\n",
		},
		{
			name:      "exception removal",
			required:  []string{"A", "B", "C"},
			features:  []string{"A", "C"},
			exception: []string{"B"},
			ok:        true,
			out:       "~B\n",
		},
		{
			// http://golang.org/issue/4303
			name: "contexts reconverging",
			required: []string{
				"A",
				"pkg syscall (darwin-386), type RawSockaddrInet6 struct",
				"pkg syscall (darwin-amd64), type RawSockaddrInet6 struct",
			},
			features: []string{
				"A",
				"pkg syscall, type RawSockaddrInet6 struct",
			},
			ok:  true,
			out: "+pkg syscall, type RawSockaddrInet6 struct\n",
		},
	}
	for _, tt := range tests {
		buf := new(bytes.Buffer)
		gotok := compareAPI(buf, tt.features, tt.required, tt.optional, tt.exception)
		if gotok != tt.ok {
			t.Errorf("%s: ok = %v; want %v", tt.name, gotok, tt.ok)
		}
		if got := buf.String(); got != tt.out {
			t.Errorf("%s: output differs\nGOT:\n%s\nWANT:\n%s", tt.name, got, tt.out)
		}
	}
}
