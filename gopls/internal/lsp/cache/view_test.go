// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package cache

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"golang.org/x/tools/gopls/internal/lsp/fake"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/gopls/internal/vulncheck"
)

func TestCaseInsensitiveFilesystem(t *testing.T) {
	base := t.TempDir()

	inner := filepath.Join(base, "a/B/c/DEFgh")
	if err := os.MkdirAll(inner, 0777); err != nil {
		t.Fatal(err)
	}
	file := filepath.Join(inner, "f.go")
	if err := os.WriteFile(file, []byte("hi"), 0777); err != nil {
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

func TestFindWorkspaceModFile(t *testing.T) {
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
	}{
		{"", ""}, // no module at root, and more than one nested module
		{"a", "a/go.mod"},
		{"a/x", "a/go.mod"},
		{"a/x/y", "a/go.mod"},
		{"b/c", "b/c/go.mod"},
		{"d", "d/e/go.mod"},
		{"d/e", "d/e/go.mod"},
		{"f", "f/g/go.mod"},
	}

	for _, test := range tests {
		ctx := context.Background()
		rel := fake.RelativeTo(dir)
		folderURI := span.URIFromPath(rel.AbsPath(test.folder))
		excludeNothing := func(string) bool { return false }
		got, err := findWorkspaceModFile(ctx, folderURI, New(nil), excludeNothing)
		if err != nil {
			t.Fatal(err)
		}
		want := span.URI("")
		if test.want != "" {
			want = span.URIFromPath(rel.AbsPath(test.want))
		}
		if got != want {
			t.Errorf("findWorkspaceModFile(%q) = %q, want %q", test.folder, got, want)
		}
	}
}

func TestInVendor(t *testing.T) {
	for _, tt := range []struct {
		path     string
		inVendor bool
	}{
		{"foo/vendor/x.go", false},
		{"foo/vendor/x/x.go", true},
		{"foo/x.go", false},
		{"foo/vendor/foo.txt", false},
		{"foo/vendor/modules.txt", false},
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
		filterer := source.NewFilterer(tt.filters)
		for _, inc := range tt.included {
			if pathExcludedByFilter(inc, filterer) {
				t.Errorf("filters %q excluded %v, wanted included", tt.filters, inc)
			}
		}
		for _, exc := range tt.excluded {
			if !pathExcludedByFilter(exc, filterer) {
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

func TestView_Vulnerabilities(t *testing.T) {
	// TODO(hyangah): use t.Cleanup when we get rid of go1.13 legacy CI.
	defer func() { timeNow = time.Now }()

	now := time.Now()

	view := &View{
		vulns: make(map[span.URI]*vulncheck.Result),
	}
	file1, file2 := span.URIFromPath("f1/go.mod"), span.URIFromPath("f2/go.mod")

	vuln1 := &vulncheck.Result{AsOf: now.Add(-(maxGovulncheckResultAge * 3) / 4)} // already ~3/4*maxGovulncheckResultAge old
	view.SetVulnerabilities(file1, vuln1)

	vuln2 := &vulncheck.Result{AsOf: now} // fresh.
	view.SetVulnerabilities(file2, vuln2)

	t.Run("fresh", func(t *testing.T) {
		got := view.Vulnerabilities()
		want := map[span.URI]*vulncheck.Result{
			file1: vuln1,
			file2: vuln2,
		}

		if diff := cmp.Diff(toJSON(want), toJSON(got)); diff != "" {
			t.Errorf("view.Vulnerabilities() mismatch (-want +got):\n%s", diff)
		}
	})

	// maxGovulncheckResultAge/2 later
	timeNow = func() time.Time { return now.Add(maxGovulncheckResultAge / 2) }
	t.Run("after30min", func(t *testing.T) {
		got := view.Vulnerabilities()
		want := map[span.URI]*vulncheck.Result{
			file1: nil, // expired.
			file2: vuln2,
		}

		if diff := cmp.Diff(toJSON(want), toJSON(got)); diff != "" {
			t.Errorf("view.Vulnerabilities() mismatch (-want +got):\n%s", diff)
		}
	})

	// maxGovulncheckResultAge later
	timeNow = func() time.Time { return now.Add(maxGovulncheckResultAge + time.Minute) }

	t.Run("after1hr", func(t *testing.T) {
		got := view.Vulnerabilities()
		want := map[span.URI]*vulncheck.Result{
			file1: nil,
			file2: nil,
		}

		if diff := cmp.Diff(toJSON(want), toJSON(got)); diff != "" {
			t.Errorf("view.Vulnerabilities() mismatch (-want +got):\n%s", diff)
		}
	})
}

func toJSON(x interface{}) string {
	b, _ := json.MarshalIndent(x, "", " ")
	return string(b)
}

func TestIgnoreFilter(t *testing.T) {
	tests := []struct {
		dirs []string
		path string
		want bool
	}{
		{[]string{"a"}, "a/testdata/foo", true},
		{[]string{"a"}, "a/_ignore/foo", true},
		{[]string{"a"}, "a/.ignore/foo", true},
		{[]string{"a"}, "b/testdata/foo", false},
		{[]string{"a"}, "testdata/foo", false},
		{[]string{"a", "b"}, "b/testdata/foo", true},
		{[]string{"a"}, "atestdata/foo", false},
	}

	for _, test := range tests {
		// convert to filepaths, for convenience
		for i, dir := range test.dirs {
			test.dirs[i] = filepath.FromSlash(dir)
		}
		test.path = filepath.FromSlash(test.path)

		f := newIgnoreFilter(test.dirs)
		if got := f.ignored(test.path); got != test.want {
			t.Errorf("newIgnoreFilter(%q).ignore(%q) = %t, want %t", test.dirs, test.path, got, test.want)
		}
	}
}
