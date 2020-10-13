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
-- d/e/go.mod
module de
`
	dir, err := fake.Tempdir(workspace)
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	tests := []struct {
		folder, want string
	}{
		// no module at root.
		{"", ""},
		{"a", "a"},
		{"a/x", "a"},
		{"b/c", "b/c"},
		{"d", "d"},
		{"d/e", "d"},
	}

	for _, test := range tests {
		ctx := context.Background()
		rel := fake.RelativeTo(dir)
		folderURI := span.URIFromPath(rel.AbsPath(test.folder))
		got, err := findWorkspaceRoot(ctx, folderURI, osFileSource{})
		if err != nil {
			t.Fatal(err)
		}
		if rel.RelPath(got.Filename()) != test.want {
			t.Errorf("fileWorkspaceRoot(%q) = %q, want %q", test.folder, got, test.want)
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
