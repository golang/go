// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build cgo

package packages_test

import (
	"runtime"
	"strings"
	"testing"

	"golang.org/x/tools/go/packages"
)

func TestLoadImportsC(t *testing.T) {
	// This test checks that when a package depends on the
	// test variant of "syscall", "unsafe", or "runtime/cgo", that dependency
	// is not removed when those packages are added when it imports "C".
	//
	// For this test to work, the external test of syscall must have a dependency
	// on net, and net must import "syscall" and "C".
	if runtime.GOOS == "windows" {
		t.Skipf("skipping on windows; packages on windows do not satisfy conditions for test.")
	}
	if runtime.GOOS == "plan9" {
		// See https://golang.org/issue/27100.
		t.Skip(`skipping on plan9; for some reason "net [syscall.test]" is not loaded`)
	}

	cfg := &packages.Config{
		Mode:  packages.LoadImports,
		Tests: true,
	}
	initial, err := packages.Load(cfg, "syscall", "net")
	if err != nil {
		t.Fatalf("failed to load imports: %v", err)
	}

	_, all := importGraph(initial)

	for _, test := range []struct {
		pattern    string
		wantImport string // an import to check for
	}{
		{"net", "syscall:syscall"},
		{"net [syscall.test]", "syscall:syscall [syscall.test]"},
		{"syscall_test [syscall.test]", "net:net [syscall.test]"},
	} {
		// Test the import paths.
		pkg := all[test.pattern]
		if pkg == nil {
			t.Errorf("package %q not loaded", test.pattern)
			continue
		}
		if imports := strings.Join(imports(pkg), " "); !strings.Contains(imports, test.wantImport) {
			t.Errorf("package %q: got \n%s, \nwant to have %s", test.pattern, imports, test.wantImport)
		}
	}
}
