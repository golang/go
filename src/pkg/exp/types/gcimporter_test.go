// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"exec"
	"go/ast"
	"io/ioutil"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

var gcName, gcPath string // compiler name and path

func init() {
	// determine compiler
	switch runtime.GOARCH {
	case "386":
		gcName = "8g"
	case "amd64":
		gcName = "6g"
	case "arm":
		gcName = "5g"
	default:
		gcName = "unknown-GOARCH-compiler"
		gcPath = gcName
		return
	}
	gcPath, _ = exec.LookPath(gcName)
}

func compile(t *testing.T, dirname, filename string) {
	cmd := exec.Command(gcPath, filename)
	cmd.Dir = dirname
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Errorf("%s %s failed: %s", gcName, filename, err)
		return
	}
	t.Logf("%s", string(out))
}

// Use the same global imports map for all tests. The effect is
// as if all tested packages were imported into a single package.
var imports = make(map[string]*ast.Object)

func testPath(t *testing.T, path string) bool {
	_, err := GcImporter(imports, path)
	if err != nil {
		t.Errorf("testPath(%s): %s", path, err)
		return false
	}
	return true
}

const maxTime = 3e9 // maximum allotted testing time in ns

func testDir(t *testing.T, dir string, endTime int64) (nimports int) {
	dirname := filepath.Join(pkgRoot, dir)
	list, err := ioutil.ReadDir(dirname)
	if err != nil {
		t.Errorf("testDir(%s): %s", dirname, err)
	}
	for _, f := range list {
		if time.Nanoseconds() >= endTime {
			t.Log("testing time used up")
			return
		}
		switch {
		case f.IsRegular():
			// try extensions
			for _, ext := range pkgExts {
				if strings.HasSuffix(f.Name, ext) {
					name := f.Name[0 : len(f.Name)-len(ext)] // remove extension
					if testPath(t, filepath.Join(dir, name)) {
						nimports++
					}
				}
			}
		case f.IsDirectory():
			nimports += testDir(t, filepath.Join(dir, f.Name), endTime)
		}
	}
	return
}

func TestGcImport(t *testing.T) {
	compile(t, "testdata", "exports.go")

	nimports := 0
	if testPath(t, "./testdata/exports") {
		nimports++
	}
	nimports += testDir(t, "", time.Nanoseconds()+maxTime) // installed packages
	t.Logf("tested %d imports", nimports)
}
