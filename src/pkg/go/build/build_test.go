// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package build

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

var buildDirs = []string{
	"pkg/path",
	"cmd/gofix",
	"pkg/big",
	"pkg/go/build/cgotest",
}

func TestBuild(t *testing.T) {
	out, err := filepath.Abs("_test/out")
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range buildDirs {
		if runtime.GOARCH == "arm" && strings.Contains(d, "/cgo") {
			// no cgo for arm, yet.
			continue
		}
		dir := filepath.Join(runtime.GOROOT(), "src", d)
		testBuild(t, dir, out)
	}
}

func testBuild(t *testing.T, dir, targ string) {
	d, err := ScanDir(dir, true)
	if err != nil {
		t.Error(err)
		return
	}
	defer os.Remove(targ)
	cmds, err := d.Build(targ)
	if err != nil {
		t.Error(err)
		return
	}
	for _, c := range cmds {
		t.Log("Run:", c)
		err = c.Run(dir)
		if err != nil {
			t.Error(c, err)
			return
		}
	}
}
