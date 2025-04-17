// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package objabi

import (
	"internal/testenv"
	"os/exec"
	"strings"
	"testing"
)

var escapeTests = []struct {
	Path    string
	Escaped string
}{
	{"foo/bar/v1", "foo/bar/v1"},
	{"foo/bar/v.1", "foo/bar/v%2e1"},
	{"f.o.o/b.a.r/v1", "f.o.o/b.a.r/v1"},
	{"f.o.o/b.a.r/v.1", "f.o.o/b.a.r/v%2e1"},
	{"f.o.o/b.a.r/v..1", "f.o.o/b.a.r/v%2e%2e1"},
	{"f.o.o/b.a.r/v..1.", "f.o.o/b.a.r/v%2e%2e1%2e"},
	{"f.o.o/b.a.r/v%1", "f.o.o/b.a.r/v%251"},
	{"runtime", "runtime"},
	{"sync/atomic", "sync/atomic"},
	{"golang.org/x/tools/godoc", "golang.org/x/tools/godoc"},
	{"foo.bar/baz.quux", "foo.bar/baz%2equux"},
	{"", ""},
	{"%foo%bar", "%25foo%25bar"},
	{"\x01\x00\x7F☺", "%01%00%7f%e2%98%ba"},
}

func TestPathToPrefix(t *testing.T) {
	for _, tc := range escapeTests {
		if got := PathToPrefix(tc.Path); got != tc.Escaped {
			t.Errorf("expected PathToPrefix(%s) = %s, got %s", tc.Path, tc.Escaped, got)
		}
	}
}

func TestPrefixToPath(t *testing.T) {
	for _, tc := range escapeTests {
		got, err := PrefixToPath(tc.Escaped)
		if err != nil {
			t.Errorf("expected PrefixToPath(%s) err = nil, got %v", tc.Escaped, err)
		}
		if got != tc.Path {
			t.Errorf("expected PrefixToPath(%s) = %s, got %s", tc.Escaped, tc.Path, got)
		}
	}
}

func TestPrefixToPathError(t *testing.T) {
	tests := []string{
		"foo%",
		"foo%1",
		"foo%%12",
		"foo%1g",
	}
	for _, tc := range tests {
		_, err := PrefixToPath(tc)
		if err == nil {
			t.Errorf("expected PrefixToPath(%s) err != nil, got nil", tc)
		}
	}
}

func TestRuntimePackageList(t *testing.T) {
	t.Skip("TODO: XXX")
	// Test that all packages imported by the runtime are marked as runtime
	// packages.
	testenv.MustHaveGoBuild(t)
	goCmd, err := testenv.GoTool()
	if err != nil {
		t.Fatal(err)
	}
	pkgList, err := exec.Command(goCmd, "list", "-deps", "runtime").Output()
	if err != nil {
		if err, ok := err.(*exec.ExitError); ok {
			t.Log(string(err.Stderr))
		}
		t.Fatal(err)
	}
	for _, pkg := range strings.Split(strings.TrimRight(string(pkgList), "\n"), "\n") {
		if pkg == "unsafe" {
			continue
		}
		if !LookupPkgSpecial(pkg).Runtime {
			t.Errorf("package %s is imported by runtime, but not marked Runtime", pkg)
		}
	}
}
