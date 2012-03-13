// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package build

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestMatch(t *testing.T) {
	ctxt := Default
	what := "default"
	match := func(tag string) {
		if !ctxt.match(tag) {
			t.Errorf("%s context should match %s, does not", what, tag)
		}
	}
	nomatch := func(tag string) {
		if ctxt.match(tag) {
			t.Errorf("%s context should NOT match %s, does", what, tag)
		}
	}

	match(runtime.GOOS + "," + runtime.GOARCH)
	match(runtime.GOOS + "," + runtime.GOARCH + ",!foo")
	nomatch(runtime.GOOS + "," + runtime.GOARCH + ",foo")

	what = "modified"
	ctxt.BuildTags = []string{"foo"}
	match(runtime.GOOS + "," + runtime.GOARCH)
	match(runtime.GOOS + "," + runtime.GOARCH + ",foo")
	nomatch(runtime.GOOS + "," + runtime.GOARCH + ",!foo")
	match(runtime.GOOS + "," + runtime.GOARCH + ",!bar")
	nomatch(runtime.GOOS + "," + runtime.GOARCH + ",bar")
	nomatch("!")
}

func TestDotSlashImport(t *testing.T) {
	p, err := ImportDir("testdata/other", 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Imports) != 1 || p.Imports[0] != "./file" {
		t.Fatalf("testdata/other: Imports=%v, want [./file]", p.Imports)
	}

	p1, err := Import("./file", "testdata/other", 0)
	if err != nil {
		t.Fatal(err)
	}
	if p1.Name != "file" {
		t.Fatalf("./file: Name=%q, want %q", p1.Name, "file")
	}
	dir := filepath.Clean("testdata/other/file") // Clean to use \ on Windows
	if p1.Dir != dir {
		t.Fatalf("./file: Dir=%q, want %q", p1.Name, dir)
	}
}

func TestLocalDirectory(t *testing.T) {
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}

	p, err := ImportDir(cwd, 0)
	if err != nil {
		t.Fatal(err)
	}
	if p.ImportPath != "go/build" {
		t.Fatalf("ImportPath=%q, want %q", p.ImportPath, "go/build")
	}
}
