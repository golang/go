// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vcs

import (
	"io/ioutil"
	"os"
	pathpkg "path"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

// Test that RepoRootForImportPath creates the correct RepoRoot for a given importPath.
// TODO(cmang): Add tests for SVN and BZR.
func TestRepoRootForImportPath(t *testing.T) {
	if runtime.GOOS == "android" {
		t.Skipf("incomplete source tree on %s", runtime.GOOS)
	}

	tests := []struct {
		path string
		want *RepoRoot
	}{
		{
			"github.com/golang/groupcache",
			&RepoRoot{
				VCS:  vcsGit,
				Repo: "https://github.com/golang/groupcache",
			},
		},
	}

	for _, test := range tests {
		got, err := RepoRootForImportPath(test.path, false)
		if err != nil {
			t.Errorf("RepoRootForImport(%q): %v", test.path, err)
			continue
		}
		want := test.want
		if got.VCS.Name != want.VCS.Name || got.Repo != want.Repo {
			t.Errorf("RepoRootForImport(%q) = VCS(%s) Repo(%s), want VCS(%s) Repo(%s)", test.path, got.VCS, got.Repo, want.VCS, want.Repo)
		}
	}
}

// Test that FromDir correctly inspects a given directory and returns the right VCS and root.
func TestFromDir(t *testing.T) {
	type testStruct struct {
		path string
		want *RepoRoot
	}

	tests := make([]testStruct, len(vcsList))
	tempDir, err := ioutil.TempDir("", "vcstest")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tempDir)

	for i, vcs := range vcsList {
		tests[i] = testStruct{
			path: filepath.Join(tempDir, "example.com", vcs.Name, "."+vcs.Cmd),
			want: &RepoRoot{
				VCS:  vcs,
				Root: pathpkg.Join("example.com", vcs.Name),
			},
		}
	}

	for _, test := range tests {
		os.MkdirAll(test.path, 0755)
		var (
			got = new(RepoRoot)
			err error
		)
		got.VCS, got.Root, err = FromDir(test.path, tempDir)
		if err != nil {
			t.Errorf("FromDir(%q, %q): %v", test.path, tempDir, err)
			os.RemoveAll(test.path)
			continue
		}
		want := test.want
		if got.VCS.Name != want.VCS.Name || got.Root != want.Root {
			t.Errorf("FromDir(%q, %q) = VCS(%s) Root(%s), want VCS(%s) Root(%s)", test.path, tempDir, got.VCS, got.Root, want.VCS, want.Root)
		}
		os.RemoveAll(test.path)
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
