// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package cache

import (
	"context"
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	"golang.org/x/tools/internal/lsp/fake"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

func TestCaseInsensitiveFilesystem(t *testing.T) {
	base, err := ioutil.TempDir("", t.Name())
	if err != nil {
		t.Fatal(err)
	}

	inner := filepath.Join(base, "a/B/c/DEFgh")
	if err := os.MkdirAll(inner, 0777); err != nil {
		t.Fatal(err)
	}
	file := filepath.Join(inner, "f.go")
	if err := ioutil.WriteFile(file, []byte("hi"), 0777); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(inner, "F.go")); err != nil {
		t.Skip("filesystem is case-sensitive")
	}

	tests := []struct {
		path string
		err  bool
	}{
		{file, false},
		{filepath.Join(inner, "F.go"), true},
		{filepath.Join(base, "a/b/c/defgh/f.go"), true},
	}
	for _, tt := range tests {
		err := checkPathCase(tt.path)
		if err != nil != tt.err {
			t.Errorf("checkPathCase(%q) = %v, wanted error: %v", tt.path, err, tt.err)
		}
	}
}

func TestFindWorkspaceRoot(t *testing.T) {
	workspace := `
-- a/go.mod --
module a
-- a/x/x.go
package x
-- a/x/y/y.go
package x
-- b/go.mod --
module b
-- b/c/go.mod --
module bc
-- d/gopls.mod --
module d-goplsworkspace
-- d/e/go.mod --
module de
-- f/g/go.mod --
module fg
`
	dir, err := fake.Tempdir(fake.UnpackTxt(workspace))
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	tests := []struct {
		folder, want string
		experimental bool
	}{
		{"", "", false}, // no module at root, and more than one nested module
		{"a", "a", false},
		{"a/x", "a", false},
		{"a/x/y", "a", false},
		{"b/c", "b/c", false},
		{"d", "d/e", false},
		{"d", "d", true},
		{"d/e", "d/e", false},
		{"d/e", "d", true},
		{"f", "f/g", false},
		{"f", "f", true},
	}

	for _, test := range tests {
		ctx := context.Background()
		rel := fake.RelativeTo(dir)
		folderURI := span.URIFromPath(rel.AbsPath(test.folder))
		excludeNothing := func(string) bool { return false }
		got, err := findWorkspaceRoot(ctx, folderURI, &osFileSource{}, excludeNothing, test.experimental)
		if err != nil {
			t.Fatal(err)
		}
		if gotf, wantf := filepath.Clean(got.Filename()), rel.AbsPath(test.want); gotf != wantf {
			t.Errorf("findWorkspaceRoot(%q, %t) = %q, want %q", test.folder, test.experimental, gotf, wantf)
		}
	}
}

func TestInVendor(t *testing.T) {
	for _, tt := range []struct {
		path     string
		inVendor bool
	}{
		{
			path:     "foo/vendor/x.go",
			inVendor: false,
		},
		{
			path:     "foo/vendor/x/x.go",
			inVendor: true,
		},
		{
			path:     "foo/x.go",
			inVendor: false,
		},
	} {
		if got := inVendor(span.URIFromPath(tt.path)); got != tt.inVendor {
			t.Errorf("expected %s inVendor %v, got %v", tt.path, tt.inVendor, got)
		}
	}
}

func TestFilters(t *testing.T) {
	tests := []struct {
		filters  []string
		included []string
		excluded []string
	}{
		{
			included: []string{"x"},
		},
		{
			filters:  []string{"-"},
			excluded: []string{"x", "x/a"},
		},
		{
			filters:  []string{"-x", "+y"},
			included: []string{"y", "y/a", "z"},
			excluded: []string{"x", "x/a"},
		},
		{
			filters:  []string{"-x", "+x/y", "-x/y/z"},
			included: []string{"x/y", "x/y/a", "a"},
			excluded: []string{"x", "x/a", "x/y/z/a"},
		},
		{
			filters:  []string{"+foobar", "-foo"},
			included: []string{"foobar", "foobar/a"},
			excluded: []string{"foo", "foo/a"},
		},
	}

	for _, tt := range tests {
		opts := &source.Options{}
		opts.DirectoryFilters = tt.filters
		for _, inc := range tt.included {
			if pathExcludedByFilter(inc, "root", "root/gopath/pkg/mod", opts) {
				t.Errorf("filters %q excluded %v, wanted included", tt.filters, inc)
			}
		}
		for _, exc := range tt.excluded {
			if !pathExcludedByFilter(exc, "root", "root/gopath/pkg/mod", opts) {
				t.Errorf("filters %q included %v, wanted excluded", tt.filters, exc)
			}
		}
	}
}

func TestSuffixes(t *testing.T) {
	type file struct {
		path string
		want bool
	}
	type cases struct {
		option []string
		files  []file
	}
	tests := []cases{
		{[]string{"tmpl", "gotmpl"}, []file{ // default
			{"foo", false},
			{"foo.tmpl", true},
			{"foo.gotmpl", true},
			{"tmpl", false},
			{"tmpl.go", false}},
		},
		{[]string{"tmpl", "gotmpl", "html", "gohtml"}, []file{
			{"foo.gotmpl", true},
			{"foo.html", true},
			{"foo.gohtml", true},
			{"html", false}},
		},
		{[]string{"tmpl", "gotmpl", ""}, []file{ // possible user mistake
			{"foo.gotmpl", true},
			{"foo.go", false},
			{"foo", false}},
		},
	}
	for _, a := range tests {
		suffixes := a.option
		for _, b := range a.files {
			got := fileHasExtension(b.path, suffixes)
			if got != b.want {
				t.Errorf("got %v, want %v, option %q, file %q (%+v)",
					got, b.want, a.option, b.path, b)
			}
		}
	}
}
