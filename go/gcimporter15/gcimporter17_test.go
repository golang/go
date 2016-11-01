// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.7,!go1.8

package gcimporter

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

func compileNewExport(t *testing.T, dirname, filename string) string {
	/* testenv. */ MustHaveGoBuild(t)
	cmd := exec.Command("go", "tool", "compile", "-newexport", filename)
	cmd.Dir = dirname
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Logf("%s", out)
		t.Fatalf("go tool compile %s failed: %s", filename, err)
	}
	// filename should end with ".go"
	return filepath.Join(dirname, filename[:len(filename)-2]+"o")
}

func TestImportTestdataNewExport(t *testing.T) {
	// This package only handles gc export data.
	if runtime.Compiler != "gc" {
		t.Skipf("gc-built packages not available (compiler = %s)", runtime.Compiler)
		return
	}

	if outFn := compileNewExport(t, "testdata", testfile); outFn != "" {
		defer os.Remove(outFn)
	}

	// filename should end with ".go"
	filename := testfile[:len(testfile)-3]
	if pkg := testPath(t, "./testdata/"+filename, "."); pkg != nil {
		// The package's Imports list must include all packages
		// explicitly imported by testfile, plus all packages
		// referenced indirectly via exported objects in testfile.
		want := `[package ast ("go/ast") package token ("go/token")]`
		got := fmt.Sprint(pkg.Imports())
		if got != want {
			t.Errorf(`Package("exports").Imports() = %s, want %s`, got, want)
		}
	}
}
