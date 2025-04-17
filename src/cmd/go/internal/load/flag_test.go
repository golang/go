// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import (
	"fmt"
	"path/filepath"
	"reflect"
	"testing"
)

type ppfTestPackage struct {
	path    string
	dir     string
	cmdline bool
	flags   []string
}

type ppfTest struct {
	args []string
	pkgs []ppfTestPackage
}

var ppfTests = []ppfTest{
	// -gcflags=-S applies only to packages on command line.
	{
		args: []string{"-S"},
		pkgs: []ppfTestPackage{
			{cmdline: true, flags: []string{"-S"}},
			{cmdline: false, flags: []string{}},
		},
	},

	// -gcflags=-S -gcflags= overrides the earlier -S.
	{
		args: []string{"-S", ""},
		pkgs: []ppfTestPackage{
			{cmdline: true, flags: []string{}},
		},
	},

	// -gcflags=net=-S applies only to package net
	{
		args: []string{"net=-S"},
		pkgs: []ppfTestPackage{
			{path: "math", cmdline: true, flags: []string{}},
			{path: "net", flags: []string{"-S"}},
		},
	},

	// -gcflags=net=-S -gcflags=net= also overrides the earlier -S
	{
		args: []string{"net=-S", "net="},
		pkgs: []ppfTestPackage{
			{path: "net", flags: []string{}},
		},
	},

	// -gcflags=net/...=-S net math
	// applies -S to net and net/http but not math
	{
		args: []string{"net/...=-S"},
		pkgs: []ppfTestPackage{
			{path: "net", flags: []string{"-S"}},
			{path: "net/http", flags: []string{"-S"}},
			{path: "math", flags: []string{}},
		},
	},

	// -gcflags=net/...=-S -gcflags=-m net math
	// applies -m to net and math and -S to other packages matching net/...
	// (net matches too, but it was grabbed by the later -gcflags).
	{
		args: []string{"net/...=-S", "-m"},
		pkgs: []ppfTestPackage{
			{path: "net", cmdline: true, flags: []string{"-m"}},
			{path: "math", cmdline: true, flags: []string{"-m"}},
			{path: "net", cmdline: false, flags: []string{"-S"}},
			{path: "net/http", flags: []string{"-S"}},
			{path: "math", flags: []string{}},
		},
	},

	// relative path patterns
	// ppfDirTest(pattern, n, dirs...) says the first n dirs should match and the others should not.
	ppfDirTest(".", 1, "/my/test/dir", "/my/test", "/my/test/other", "/my/test/dir/sub"),
	ppfDirTest("..", 1, "/my/test", "/my/test/dir", "/my/test/other", "/my/test/dir/sub"),
	ppfDirTest("./sub", 1, "/my/test/dir/sub", "/my/test", "/my/test/dir", "/my/test/other", "/my/test/dir/sub/sub"),
	ppfDirTest("../other", 1, "/my/test/other", "/my/test", "/my/test/dir", "/my/test/other/sub", "/my/test/dir/other", "/my/test/dir/sub"),
	ppfDirTest("./...", 3, "/my/test/dir", "/my/test/dir/sub", "/my/test/dir/sub/sub", "/my/test/other", "/my/test/other/sub"),
	ppfDirTest("../...", 4, "/my/test/dir", "/my/test/other", "/my/test/dir/sub", "/my/test/other/sub", "/my/other/test"),
	ppfDirTest("../...sub...", 3, "/my/test/dir/sub", "/my/test/othersub", "/my/test/yellowsubmarine", "/my/other/test"),
}

func ppfDirTest(pattern string, nmatch int, dirs ...string) ppfTest {
	var pkgs []ppfTestPackage
	for i, d := range dirs {
		flags := []string{}
		if i < nmatch {
			flags = []string{"-S"}
		}
		pkgs = append(pkgs, ppfTestPackage{path: "p", dir: d, flags: flags})
	}
	return ppfTest{args: []string{pattern + "=-S"}, pkgs: pkgs}
}

func TestPerPackageFlag(t *testing.T) {
	nativeDir := func(d string) string {
		if filepath.Separator == '\\' {
			return `C:` + filepath.FromSlash(d)
		}
		return d
	}

	for i, tt := range ppfTests {
		t.Run(fmt.Sprintf("#%d", i), func(t *testing.T) {
			ppFlags := new(PerPackageFlag)
			for _, arg := range tt.args {
				t.Logf("set(%s)", arg)
				if err := ppFlags.set(arg, nativeDir("/my/test/dir")); err != nil {
					t.Fatal(err)
				}
			}
			for _, p := range tt.pkgs {
				dir := nativeDir(p.dir)
				flags := ppFlags.For(&Package{PackagePublic: PackagePublic{ImportPath: p.path, Dir: dir}, Internal: PackageInternal{CmdlinePkg: p.cmdline}})
				if !reflect.DeepEqual(flags, p.flags) {
					t.Errorf("For(%v, %v, %v) = %v, want %v", p.path, dir, p.cmdline, flags, p.flags)
				}
			}
		})
	}
}
