// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

var foldDupTests = []struct {
	list   []string
	f1, f2 string
}{
	{stringList("math/rand", "math/big"), "", ""},
	{stringList("math", "strings"), "", ""},
	{stringList("strings"), "", ""},
	{stringList("strings", "strings"), "strings", "strings"},
	{stringList("Rand", "rand", "math", "math/rand", "math/Rand"), "Rand", "rand"},
}

func TestFoldDup(t *testing.T) {
	for _, tt := range foldDupTests {
		f1, f2 := foldDup(tt.list)
		if f1 != tt.f1 || f2 != tt.f2 {
			t.Errorf("foldDup(%q) = %q, %q, want %q, %q", tt.list, f1, f2, tt.f1, tt.f2)
		}
	}
}

var parseMetaGoImportsTests = []struct {
	in  string
	out []metaImport
}{
	{
		`<meta name="go-import" content="foo/bar git https://github.com/rsc/foo/bar">`,
		[]metaImport{{"foo/bar", "git", "https://github.com/rsc/foo/bar"}},
	},
	{
		`<meta name="go-import" content="foo/bar git https://github.com/rsc/foo/bar">
		<meta name="go-import" content="baz/quux git http://github.com/rsc/baz/quux">`,
		[]metaImport{
			{"foo/bar", "git", "https://github.com/rsc/foo/bar"},
			{"baz/quux", "git", "http://github.com/rsc/baz/quux"},
		},
	},
	{
		`<head>
		<meta name="go-import" content="foo/bar git https://github.com/rsc/foo/bar">
		</head>`,
		[]metaImport{{"foo/bar", "git", "https://github.com/rsc/foo/bar"}},
	},
	{
		`<meta name="go-import" content="foo/bar git https://github.com/rsc/foo/bar">
		<body>`,
		[]metaImport{{"foo/bar", "git", "https://github.com/rsc/foo/bar"}},
	},
	{
		`<!doctype html><meta name="go-import" content="foo/bar git https://github.com/rsc/foo/bar">`,
		[]metaImport{{"foo/bar", "git", "https://github.com/rsc/foo/bar"}},
	},
	{
		// XML doesn't like <div style=position:relative>.
		`<!doctype html><title>Page Not Found</title><meta name=go-import content="chitin.io/chitin git https://github.com/chitin-io/chitin"><div style=position:relative>DRAFT</div>`,
		[]metaImport{{"chitin.io/chitin", "git", "https://github.com/chitin-io/chitin"}},
	},
}

func TestParseMetaGoImports(t *testing.T) {
	for i, tt := range parseMetaGoImportsTests {
		out, err := parseMetaGoImports(strings.NewReader(tt.in))
		if err != nil {
			t.Errorf("test#%d: %v", i, err)
			continue
		}
		if !reflect.DeepEqual(out, tt.out) {
			t.Errorf("test#%d:\n\thave %q\n\twant %q", i, out, tt.out)
		}
	}
}

func TestSharedLibName(t *testing.T) {
	// TODO(avdva) - make these values platform-specific
	prefix := "lib"
	suffix := ".so"
	testData := []struct {
		args      []string
		pkgs      []*Package
		expected  string
		expectErr bool
		rootedAt  string
	}{
		{
			args:     []string{"std"},
			pkgs:     []*Package{},
			expected: "std",
		},
		{
			args:     []string{"std", "cmd"},
			pkgs:     []*Package{},
			expected: "std,cmd",
		},
		{
			args:     []string{},
			pkgs:     []*Package{&Package{ImportPath: "gopkg.in/somelib"}},
			expected: "gopkg.in-somelib",
		},
		{
			args:     []string{"./..."},
			pkgs:     []*Package{&Package{ImportPath: "somelib"}},
			expected: "somelib",
			rootedAt: "somelib",
		},
		{
			args:     []string{"../somelib", "../somelib"},
			pkgs:     []*Package{&Package{ImportPath: "somelib"}},
			expected: "somelib",
		},
		{
			args:     []string{"../lib1", "../lib2"},
			pkgs:     []*Package{&Package{ImportPath: "gopkg.in/lib1"}, &Package{ImportPath: "gopkg.in/lib2"}},
			expected: "gopkg.in-lib1,gopkg.in-lib2",
		},
		{
			args: []string{"./..."},
			pkgs: []*Package{
				&Package{ImportPath: "gopkg.in/dir/lib1"},
				&Package{ImportPath: "gopkg.in/lib2"},
				&Package{ImportPath: "gopkg.in/lib3"},
			},
			expected: "gopkg.in",
			rootedAt: "gopkg.in",
		},
		{
			args:      []string{"std", "../lib2"},
			pkgs:      []*Package{},
			expectErr: true,
		},
		{
			args:      []string{"all", "./"},
			pkgs:      []*Package{},
			expectErr: true,
		},
		{
			args:      []string{"cmd", "fmt"},
			pkgs:      []*Package{},
			expectErr: true,
		},
	}
	for _, data := range testData {
		func() {
			if data.rootedAt != "" {
				tmpGopath, err := ioutil.TempDir("", "gopath")
				if err != nil {
					t.Fatal(err)
				}
				oldGopath := buildContext.GOPATH
				defer func() {
					buildContext.GOPATH = oldGopath
					os.Chdir(cwd)
					err := os.RemoveAll(tmpGopath)
					if err != nil {
						t.Error(err)
					}
				}()
				root := filepath.Join(tmpGopath, "src", data.rootedAt)
				err = os.MkdirAll(root, 0755)
				if err != nil {
					t.Fatal(err)
				}
				buildContext.GOPATH = tmpGopath
				os.Chdir(root)
			}
			computed, err := libname(data.args, data.pkgs)
			if err != nil {
				if !data.expectErr {
					t.Errorf("libname returned an error %q, expected a name", err.Error())
				}
			} else if data.expectErr {
				t.Errorf("libname returned %q, expected an error", computed)
			} else {
				expected := prefix + data.expected + suffix
				if expected != computed {
					t.Errorf("libname returned %q, expected %q", computed, expected)
				}
			}
		}()
	}
}
