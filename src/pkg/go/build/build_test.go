// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package build

import (
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// TODO(adg): test building binaries

var buildPkgs = []string{
	"path",
	"big",
	"go/build/cgotest",
}

func TestBuild(t *testing.T) {
	for _, pkg := range buildPkgs {
		if runtime.GOARCH == "arm" && strings.Contains(pkg, "/cgo") {
			// no cgo for arm, yet.
			continue
		}
		tree := Path[0] // Goroot
		testBuild(t, tree, pkg)
	}
}

func testBuild(t *testing.T, tree *Tree, pkg string) {
	dir := filepath.Join(tree.SrcDir(), pkg)
	info, err := ScanDir(dir, true)
	if err != nil {
		t.Error(err)
		return
	}
	s, err := Build(tree, pkg, info)
	if err != nil {
		t.Error(err)
		return
	}
	for _, c := range s.Cmd {
		t.Log("Run:", c)
		err = c.Run()
		if err != nil {
			t.Error(c, err)
			return
		}
	}
}
