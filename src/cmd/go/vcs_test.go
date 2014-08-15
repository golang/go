// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"runtime"
	"testing"
)

// Test that RepoRootForImportPath creates the correct RepoRoot for a given importPath.
// TODO(cmang): Add tests for SVN and BZR.
func TestRepoRootForImportPath(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test to avoid external network")
	}
	switch runtime.GOOS {
	case "nacl", "android":
		t.Skipf("no networking available on %s", runtime.GOOS)
	}
	tests := []struct {
		path string
		want *repoRoot
	}{
		{
			"code.google.com/p/go",
			&repoRoot{
				vcs:  vcsHg,
				repo: "https://code.google.com/p/go",
			},
		},
		/*{
		        "code.google.com/r/go",
		        &repoRoot{
		                vcs:  vcsHg,
		                repo: "https://code.google.com/r/go",
		        },
		},*/
		{
			"github.com/golang/groupcache",
			&repoRoot{
				vcs:  vcsGit,
				repo: "https://github.com/golang/groupcache",
			},
		},
		// IBM DevOps Services tests
		{
			"hub.jazz.net/git/user1/pkgname",
			&repoRoot{
				vcs:  vcsGit,
				repo: "https://hub.jazz.net/git/user1/pkgname",
			},
		},
		{
			"hub.jazz.net/git/user1/pkgname/submodule/submodule/submodule",
			&repoRoot{
				vcs:  vcsGit,
				repo: "https://hub.jazz.net/git/user1/pkgname",
			},
		},
		{
			"hub.jazz.net",
			nil,
		},
		{
			"hub2.jazz.net",
			nil,
		},
		{
			"hub.jazz.net/someotherprefix",
			nil,
		},
		{
			"hub.jazz.net/someotherprefix/user1/pkgname",
			nil,
		},
		// Spaces are not valid in user names or package names
		{
			"hub.jazz.net/git/User 1/pkgname",
			nil,
		},
		{
			"hub.jazz.net/git/user1/pkg name",
			nil,
		},
		// Dots are not valid in user names
		{
			"hub.jazz.net/git/user.1/pkgname",
			nil,
		},
		{
			"hub.jazz.net/git/user/pkg.name",
			&repoRoot{
				vcs:  vcsGit,
				repo: "https://hub.jazz.net/git/user/pkg.name",
			},
		},
		// User names cannot have uppercase letters
		{
			"hub.jazz.net/git/USER/pkgname",
			nil,
		},
	}

	for _, test := range tests {
		got, err := repoRootForImportPath(test.path)
		want := test.want

		if want == nil {
			if err == nil {
				t.Errorf("RepoRootForImport(%q): Error expected but not received", test.path)
			}
			continue
		}
		if err != nil {
			t.Errorf("RepoRootForImport(%q): %v", test.path, err)
			continue
		}
		if got.vcs.name != want.vcs.name || got.repo != want.repo {
			t.Errorf("RepoRootForImport(%q) = VCS(%s) Repo(%s), want VCS(%s) Repo(%s)", test.path, got.vcs, got.repo, want.vcs, want.repo)
		}
	}
}
