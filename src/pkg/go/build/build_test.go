// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package build

import (
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"testing"
)

func sortstr(x []string) []string {
	sort.Strings(x)
	return x
}

var buildPkgs = []struct {
	dir  string
	info *DirInfo
}{
	{
		"go/build/pkgtest",
		&DirInfo{
			GoFiles:      []string{"pkgtest.go"},
			SFiles:       []string{"sqrt_" + runtime.GOARCH + ".s"},
			Package:      "pkgtest",
			Imports:      []string{"bytes"},
			TestImports:  []string{"fmt", "pkgtest"},
			TestGoFiles:  sortstr([]string{"sqrt_test.go", "sqrt_" + runtime.GOARCH + "_test.go"}),
			XTestGoFiles: []string{"xsqrt_test.go"},
		},
	},
	{
		"go/build/cmdtest",
		&DirInfo{
			GoFiles:     []string{"main.go"},
			Package:     "main",
			Imports:     []string{"go/build/pkgtest"},
			TestImports: []string{},
		},
	},
	{
		"go/build/cgotest",
		&DirInfo{
			CgoFiles:    ifCgo([]string{"cgotest.go"}),
			CFiles:      []string{"cgotest.c"},
			HFiles:      []string{"cgotest.h"},
			Imports:     []string{"C", "unsafe"},
			TestImports: []string{},
			Package:     "cgotest",
		},
	},
}

func ifCgo(x []string) []string {
	if DefaultContext.CgoEnabled {
		return x
	}
	return nil
}

func TestBuild(t *testing.T) {
	for _, tt := range buildPkgs {
		tree := Path[0] // Goroot
		dir := filepath.Join(tree.SrcDir(), tt.dir)
		info, err := ScanDir(dir)
		if err != nil {
			t.Errorf("ScanDir(%#q): %v", tt.dir, err)
			continue
		}
		if !reflect.DeepEqual(info, tt.info) {
			t.Errorf("ScanDir(%#q) = %#v, want %#v\n", tt.dir, info, tt.info)
			continue
		}
	}
}

func TestMatch(t *testing.T) {
	ctxt := DefaultContext
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
}
