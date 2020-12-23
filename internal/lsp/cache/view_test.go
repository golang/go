// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package cache

import (
	"context"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"golang.org/x/mod/modfile"
	"golang.org/x/tools/internal/lsp/fake"
	"golang.org/x/tools/internal/lsp/protocol"
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
	dir, err := fake.Tempdir(workspace)
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

// This tests the logic used to extract positions from parse and other Go
// command errors.
func TestExtractPositionFromError(t *testing.T) {
	workspace := `
-- a/go.mod --
modul a.com
-- b/go.mod --
module b.com

go 1.12.hello
-- c/go.mod --
module c.com

require a.com master
`
	dir, err := fake.Tempdir(workspace)
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	tests := []struct {
		filename string
		wantRng  protocol.Range
	}{
		{
			filename: "a/go.mod",
			wantRng:  protocol.Range{},
		},
		{
			filename: "b/go.mod",
			wantRng: protocol.Range{
				Start: protocol.Position{Line: 2},
				End:   protocol.Position{Line: 2},
			},
		},
		{
			filename: "c/go.mod",
			wantRng: protocol.Range{
				Start: protocol.Position{Line: 2},
				End:   protocol.Position{Line: 2},
			},
		},
	}
	for _, test := range tests {
		ctx := context.Background()
		rel := fake.RelativeTo(dir)
		uri := span.URIFromPath(rel.AbsPath(test.filename))
		if source.DetectLanguage("", uri.Filename()) != source.Mod {
			t.Fatalf("expected only go.mod files")
		}
		// Try directly parsing the given, invalid go.mod file. Then, extract a
		// position from the error message.
		src := &osFileSource{}
		modFH, err := src.GetFile(ctx, uri)
		if err != nil {
			t.Fatal(err)
		}
		content, err := modFH.Read()
		if err != nil {
			t.Fatal(err)
		}
		_, parseErr := modfile.Parse(uri.Filename(), content, nil)
		if parseErr == nil {
			t.Fatalf("%s: expected an unparseable go.mod file", uri.Filename())
		}
		srcErr := extractErrorWithPosition(ctx, parseErr.Error(), src)
		if srcErr == nil {
			t.Fatalf("unable to extract positions from %v", parseErr.Error())
		}
		if srcErr.URI != uri {
			t.Errorf("unexpected URI: got %s, wanted %s", srcErr.URI, uri)
		}
		if protocol.CompareRange(test.wantRng, srcErr.Range) != 0 {
			t.Errorf("unexpected range: got %s, wanted %s", srcErr.Range, test.wantRng)
		}
		if !strings.HasSuffix(parseErr.Error(), srcErr.Message) {
			t.Errorf("unexpected message: got %s, wanted %s", srcErr.Message, parseErr)
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
			if pathExcludedByFilter(inc, opts) {
				t.Errorf("filters %q excluded %v, wanted included", tt.filters, inc)
			}
		}
		for _, exc := range tt.excluded {
			if !pathExcludedByFilter(exc, opts) {
				t.Errorf("filters %q included %v, wanted excluded", tt.filters, exc)
			}
		}
	}
}
