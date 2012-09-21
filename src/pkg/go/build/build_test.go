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

func TestShouldBuild(t *testing.T) {
	const file1 = "// +build tag1\n\n" +
		"package main\n"

	const file2 = "// +build cgo\n\n" +
		"// This package implements parsing of tags like\n" +
		"// +build tag1\n" +
		"package build"

	const file3 = "// Copyright The Go Authors.\n\n" +
		"package build\n\n" +
		"// shouldBuild checks tags given by lines of the form\n" +
		"// +build tag\n" +
		"func shouldBuild(content []byte)\n"

	ctx := &Context{BuildTags: []string{"tag1"}}
	if !ctx.shouldBuild([]byte(file1)) {
		t.Errorf("should not build file1, expected the contrary")
	}
	if ctx.shouldBuild([]byte(file2)) {
		t.Errorf("should build file2, expected the contrary")
	}

	ctx = &Context{BuildTags: nil}
	if !ctx.shouldBuild([]byte(file3)) {
		t.Errorf("should not build file3, expected the contrary")
	}
}
