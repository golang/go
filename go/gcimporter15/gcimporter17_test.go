// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.7

package gcimporter

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

// TODO(gri) Remove this function once we switched to new export format by default.
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

// TODO(gri) Remove this function once we switched to new export format by default
//           (and update the comment and want list in TestImportTestdata).
func TestImportTestdataNewExport(t *testing.T) {
	// This package only handles gc export data.
	if runtime.Compiler != "gc" {
		t.Skipf("gc-built packages not available (compiler = %s)", runtime.Compiler)
		return
	}

	if outFn := compileNewExport(t, "testdata", "exports.go"); outFn != "" {
		defer os.Remove(outFn)
	}

	if pkg := testPath(t, "./testdata/exports", "."); pkg != nil {
		// The package's Imports list must include all packages
		// explicitly imported by exports.go, plus all packages
		// referenced indirectly via exported objects in exports.go.
		want := `[package ast ("go/ast") package token ("go/token")]`
		got := fmt.Sprint(pkg.Imports())
		if got != want {
			t.Errorf(`Package("exports").Imports() = %s, want %s`, got, want)
		}
	}
}
