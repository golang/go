// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"internal/testenv"
	"testing"
)

// Test that RepoRootForImportPath creates the correct RepoRoot for a given importPath.
// TODO(cmang): Add tests for SVN and BZR.
func TestRepoRootForImportPath(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

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
		// Spaces are not valid in package name
		{
			"git.apache.org/package name/path/to/lib",
			nil,
		},
		// Should have ".git" suffix
		{
			"git.apache.org/package-name/path/to/lib",
			nil,
		},
		{
			"git.apache.org/package-name.git",
			&repoRoot{
				vcs:  vcsGit,
				repo: "https://git.apache.org/package-name.git",
			},
		},
		{
			"git.apache.org/package-name_2.x.git/path/to/lib",
			&repoRoot{
				vcs:  vcsGit,
				repo: "https://git.apache.org/package-name_2.x.git",
			},
		},
	}

	for _, test := range tests {
		got, err := repoRootForImportPath(test.path, secure)
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

func TestIsSecure(t *testing.T) {
	tests := []struct {
		vcs    *vcsCmd
		url    string
		secure bool
	}{
		{vcsGit, "http://example.com/foo.git", false},
		{vcsGit, "https://example.com/foo.git", true},
		{vcsBzr, "http://example.com/foo.bzr", false},
		{vcsBzr, "https://example.com/foo.bzr", true},
		{vcsSvn, "http://example.com/svn", false},
		{vcsSvn, "https://example.com/svn", true},
		{vcsHg, "http://example.com/foo.hg", false},
		{vcsHg, "https://example.com/foo.hg", true},
		{vcsGit, "ssh://user@example.com/foo.git", true},
		{vcsGit, "user@server:path/to/repo.git", false},
		{vcsGit, "user@server:", false},
		{vcsGit, "server:repo.git", false},
		{vcsGit, "server:path/to/repo.git", false},
		{vcsGit, "example.com:path/to/repo.git", false},
		{vcsGit, "path/that/contains/a:colon/repo.git", false},
		{vcsHg, "ssh://user@example.com/path/to/repo.hg", true},
	}

	for _, test := range tests {
		secure := test.vcs.isSecure(test.url)
		if secure != test.secure {
			t.Errorf("%s isSecure(%q) = %t; want %t", test.vcs, test.url, secure, test.secure)
		}
	}
}
