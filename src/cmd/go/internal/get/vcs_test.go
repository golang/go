// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package get

import (
	"errors"
	"internal/testenv"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"testing"

	"cmd/go/internal/web"
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
			"github.com/golang/groupcache",
			&repoRoot{
				vcs:  vcsGit,
				repo: "https://github.com/golang/groupcache",
			},
		},
		// Unicode letters in directories (issue 18660).
		{
			"github.com/user/unicode/испытание",
			&repoRoot{
				vcs:  vcsGit,
				repo: "https://github.com/user/unicode",
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
		// OpenStack tests
		{
			"git.openstack.org/openstack/swift",
			&repoRoot{
				vcs:  vcsGit,
				repo: "https://git.openstack.org/openstack/swift",
			},
		},
		// Trailing .git is less preferred but included for
		// compatibility purposes while the same source needs to
		// be compilable on both old and new go
		{
			"git.openstack.org/openstack/swift.git",
			&repoRoot{
				vcs:  vcsGit,
				repo: "https://git.openstack.org/openstack/swift.git",
			},
		},
		{
			"git.openstack.org/openstack/swift/go/hummingbird",
			&repoRoot{
				vcs:  vcsGit,
				repo: "https://git.openstack.org/openstack/swift",
			},
		},
		{
			"git.openstack.org",
			nil,
		},
		{
			"git.openstack.org/openstack",
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
		got, err := repoRootForImportPath(test.path, web.Secure)
		want := test.want

		if want == nil {
			if err == nil {
				t.Errorf("repoRootForImportPath(%q): Error expected but not received", test.path)
			}
			continue
		}
		if err != nil {
			t.Errorf("repoRootForImportPath(%q): %v", test.path, err)
			continue
		}
		if got.vcs.name != want.vcs.name || got.repo != want.repo {
			t.Errorf("repoRootForImportPath(%q) = VCS(%s) Repo(%s), want VCS(%s) Repo(%s)", test.path, got.vcs, got.repo, want.vcs, want.repo)
		}
	}
}

// Test that vcsFromDir correctly inspects a given directory and returns the right VCS and root.
func TestFromDir(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "vcstest")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tempDir)

	for j, vcs := range vcsList {
		dir := filepath.Join(tempDir, "example.com", vcs.name, "."+vcs.cmd)
		if j&1 == 0 {
			err := os.MkdirAll(dir, 0755)
			if err != nil {
				t.Fatal(err)
			}
		} else {
			err := os.MkdirAll(filepath.Dir(dir), 0755)
			if err != nil {
				t.Fatal(err)
			}
			f, err := os.Create(dir)
			if err != nil {
				t.Fatal(err)
			}
			f.Close()
		}

		want := repoRoot{
			vcs:  vcs,
			root: path.Join("example.com", vcs.name),
		}
		var got repoRoot
		got.vcs, got.root, err = vcsFromDir(dir, tempDir)
		if err != nil {
			t.Errorf("FromDir(%q, %q): %v", dir, tempDir, err)
			continue
		}
		if got.vcs.name != want.vcs.name || got.root != want.root {
			t.Errorf("FromDir(%q, %q) = VCS(%s) Root(%s), want VCS(%s) Root(%s)", dir, tempDir, got.vcs, got.root, want.vcs, want.root)
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

func TestIsSecureGitAllowProtocol(t *testing.T) {
	tests := []struct {
		vcs    *vcsCmd
		url    string
		secure bool
	}{
		// Same as TestIsSecure to verify same behavior.
		{vcsGit, "http://example.com/foo.git", false},
		{vcsGit, "https://example.com/foo.git", true},
		{vcsBzr, "http://example.com/foo.bzr", false},
		{vcsBzr, "https://example.com/foo.bzr", true},
		{vcsSvn, "http://example.com/svn", false},
		{vcsSvn, "https://example.com/svn", true},
		{vcsHg, "http://example.com/foo.hg", false},
		{vcsHg, "https://example.com/foo.hg", true},
		{vcsGit, "user@server:path/to/repo.git", false},
		{vcsGit, "user@server:", false},
		{vcsGit, "server:repo.git", false},
		{vcsGit, "server:path/to/repo.git", false},
		{vcsGit, "example.com:path/to/repo.git", false},
		{vcsGit, "path/that/contains/a:colon/repo.git", false},
		{vcsHg, "ssh://user@example.com/path/to/repo.hg", true},
		// New behavior.
		{vcsGit, "ssh://user@example.com/foo.git", false},
		{vcsGit, "foo://example.com/bar.git", true},
		{vcsHg, "foo://example.com/bar.hg", false},
		{vcsSvn, "foo://example.com/svn", false},
		{vcsBzr, "foo://example.com/bar.bzr", false},
	}

	defer os.Unsetenv("GIT_ALLOW_PROTOCOL")
	os.Setenv("GIT_ALLOW_PROTOCOL", "https:foo")
	for _, test := range tests {
		secure := test.vcs.isSecure(test.url)
		if secure != test.secure {
			t.Errorf("%s isSecure(%q) = %t; want %t", test.vcs, test.url, secure, test.secure)
		}
	}
}

func TestMatchGoImport(t *testing.T) {
	tests := []struct {
		imports []metaImport
		path    string
		mi      metaImport
		err     error
	}{
		{
			imports: []metaImport{
				{Prefix: "example.com/user/foo", VCS: "git", RepoRoot: "https://example.com/repo/target"},
			},
			path: "example.com/user/foo",
			mi:   metaImport{Prefix: "example.com/user/foo", VCS: "git", RepoRoot: "https://example.com/repo/target"},
		},
		{
			imports: []metaImport{
				{Prefix: "example.com/user/foo", VCS: "git", RepoRoot: "https://example.com/repo/target"},
			},
			path: "example.com/user/foo/",
			mi:   metaImport{Prefix: "example.com/user/foo", VCS: "git", RepoRoot: "https://example.com/repo/target"},
		},
		{
			imports: []metaImport{
				{Prefix: "example.com/user/foo", VCS: "git", RepoRoot: "https://example.com/repo/target"},
				{Prefix: "example.com/user/fooa", VCS: "git", RepoRoot: "https://example.com/repo/target"},
			},
			path: "example.com/user/foo",
			mi:   metaImport{Prefix: "example.com/user/foo", VCS: "git", RepoRoot: "https://example.com/repo/target"},
		},
		{
			imports: []metaImport{
				{Prefix: "example.com/user/foo", VCS: "git", RepoRoot: "https://example.com/repo/target"},
				{Prefix: "example.com/user/fooa", VCS: "git", RepoRoot: "https://example.com/repo/target"},
			},
			path: "example.com/user/fooa",
			mi:   metaImport{Prefix: "example.com/user/fooa", VCS: "git", RepoRoot: "https://example.com/repo/target"},
		},
		{
			imports: []metaImport{
				{Prefix: "example.com/user/foo", VCS: "git", RepoRoot: "https://example.com/repo/target"},
				{Prefix: "example.com/user/foo/bar", VCS: "git", RepoRoot: "https://example.com/repo/target"},
			},
			path: "example.com/user/foo/bar",
			err:  errors.New("should not be allowed to create nested repo"),
		},
		{
			imports: []metaImport{
				{Prefix: "example.com/user/foo", VCS: "git", RepoRoot: "https://example.com/repo/target"},
				{Prefix: "example.com/user/foo/bar", VCS: "git", RepoRoot: "https://example.com/repo/target"},
			},
			path: "example.com/user/foo/bar/baz",
			err:  errors.New("should not be allowed to create nested repo"),
		},
		{
			imports: []metaImport{
				{Prefix: "example.com/user/foo", VCS: "git", RepoRoot: "https://example.com/repo/target"},
				{Prefix: "example.com/user/foo/bar", VCS: "git", RepoRoot: "https://example.com/repo/target"},
			},
			path: "example.com/user/foo/bar/baz/qux",
			err:  errors.New("should not be allowed to create nested repo"),
		},
		{
			imports: []metaImport{
				{Prefix: "example.com/user/foo", VCS: "git", RepoRoot: "https://example.com/repo/target"},
				{Prefix: "example.com/user/foo/bar", VCS: "git", RepoRoot: "https://example.com/repo/target"},
			},
			path: "example.com/user/foo/bar/baz/",
			err:  errors.New("should not be allowed to create nested repo"),
		},
		{
			imports: []metaImport{
				{Prefix: "example.com/user/foo", VCS: "git", RepoRoot: "https://example.com/repo/target"},
				{Prefix: "example.com/user/foo/bar", VCS: "git", RepoRoot: "https://example.com/repo/target"},
			},
			path: "example.com",
			err:  errors.New("pathologically short path"),
		},
		{
			imports: []metaImport{
				{Prefix: "example.com/user/foo", VCS: "git", RepoRoot: "https://example.com/repo/target"},
			},
			path: "different.example.com/user/foo",
			err:  errors.New("meta tags do not match import path"),
		},
	}

	for _, test := range tests {
		mi, err := matchGoImport(test.imports, test.path)
		if mi != test.mi {
			t.Errorf("unexpected metaImport; got %v, want %v", mi, test.mi)
		}

		got := err
		want := test.err
		if (got == nil) != (want == nil) {
			t.Errorf("unexpected error; got %v, want %v", got, want)
		}
	}
}
