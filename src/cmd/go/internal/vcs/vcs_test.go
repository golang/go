// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vcs

import (
	"errors"
	"fmt"
	"internal/testenv"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"cmd/go/internal/web"
)

func init() {
	// GOVCS defaults to public:git|hg,private:all,
	// which breaks many tests here - they can't use non-git, non-hg VCS at all!
	// Change to fully permissive.
	// The tests of the GOVCS setting itself are in ../../testdata/script/govcs.txt.
	os.Setenv("GOVCS", "*:all")
}

// Test that RepoRootForImportPath determines the correct RepoRoot for a given importPath.
// TODO(cmang): Add tests for SVN and BZR.
func TestRepoRootForImportPath(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

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
		// Unicode letters in directories are not valid.
		{
			"github.com/user/unicode/испытание",
			nil,
		},
		// IBM DevOps Services tests
		{
			"hub.jazz.net/git/user1/pkgname",
			&RepoRoot{
				VCS:  vcsGit,
				Repo: "https://hub.jazz.net/git/user1/pkgname",
			},
		},
		{
			"hub.jazz.net/git/user1/pkgname/submodule/submodule/submodule",
			&RepoRoot{
				VCS:  vcsGit,
				Repo: "https://hub.jazz.net/git/user1/pkgname",
			},
		},
		{
			"hub.jazz.net",
			nil,
		},
		{
			"hubajazz.net",
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
			&RepoRoot{
				VCS:  vcsGit,
				Repo: "https://hub.jazz.net/git/user/pkg.name",
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
			&RepoRoot{
				VCS:  vcsGit,
				Repo: "https://git.openstack.org/openstack/swift",
			},
		},
		// Trailing .git is less preferred but included for
		// compatibility purposes while the same source needs to
		// be compilable on both old and new go
		{
			"git.openstack.org/openstack/swift.git",
			&RepoRoot{
				VCS:  vcsGit,
				Repo: "https://git.openstack.org/openstack/swift.git",
			},
		},
		{
			"git.openstack.org/openstack/swift/go/hummingbird",
			&RepoRoot{
				VCS:  vcsGit,
				Repo: "https://git.openstack.org/openstack/swift",
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
			"gitbapache.org",
			nil,
		},
		{
			"git.apache.org/package-name.git",
			&RepoRoot{
				VCS:  vcsGit,
				Repo: "https://git.apache.org/package-name.git",
			},
		},
		{
			"git.apache.org/package-name_2.x.git/path/to/lib",
			&RepoRoot{
				VCS:  vcsGit,
				Repo: "https://git.apache.org/package-name_2.x.git",
			},
		},
		{
			"chiselapp.com/user/kyle/repository/fossilgg",
			&RepoRoot{
				VCS:  vcsFossil,
				Repo: "https://chiselapp.com/user/kyle/repository/fossilgg",
			},
		},
		{
			// must have a user/$name/repository/$repo path
			"chiselapp.com/kyle/repository/fossilgg",
			nil,
		},
		{
			"chiselapp.com/user/kyle/fossilgg",
			nil,
		},
		{
			"bitbucket.org/workspace/pkgname",
			&RepoRoot{
				VCS:  vcsGit,
				Repo: "https://bitbucket.org/workspace/pkgname",
			},
		},
	}

	for _, test := range tests {
		got, err := RepoRootForImportPath(test.path, IgnoreMod, web.SecureOnly)
		want := test.want

		if want == nil {
			if err == nil {
				t.Errorf("RepoRootForImportPath(%q): Error expected but not received", test.path)
			}
			continue
		}
		if err != nil {
			t.Errorf("RepoRootForImportPath(%q): %v", test.path, err)
			continue
		}
		if got.VCS.Name != want.VCS.Name || got.Repo != want.Repo {
			t.Errorf("RepoRootForImportPath(%q) = VCS(%s) Repo(%s), want VCS(%s) Repo(%s)", test.path, got.VCS, got.Repo, want.VCS, want.Repo)
		}
	}
}

// Test that vcs.FromDir correctly inspects a given directory and returns the
// right VCS and repo directory.
func TestFromDir(t *testing.T) {
	tempDir := t.TempDir()

	for _, vcs := range vcsList {
		for r, root := range vcs.RootNames {
			vcsName := fmt.Sprint(vcs.Name, r)
			dir := filepath.Join(tempDir, "example.com", vcsName, root.filename)
			if root.isDir {
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

			wantRepoDir := filepath.Dir(dir)
			gotRepoDir, gotVCS, err := FromDir(dir, tempDir, false)
			if err != nil {
				t.Errorf("FromDir(%q, %q): %v", dir, tempDir, err)
				continue
			}
			if gotRepoDir != wantRepoDir || gotVCS.Name != vcs.Name {
				t.Errorf("FromDir(%q, %q) = RepoDir(%s), VCS(%s); want RepoDir(%s), VCS(%s)", dir, tempDir, gotRepoDir, gotVCS.Name, wantRepoDir, vcs.Name)
			}
		}
	}
}

func TestIsSecure(t *testing.T) {
	tests := []struct {
		vcs    *Cmd
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
		{vcsFossil, "http://example.com/foo", false},
		{vcsFossil, "https://example.com/foo", true},
	}

	for _, test := range tests {
		secure := test.vcs.IsSecure(test.url)
		if secure != test.secure {
			t.Errorf("%s isSecure(%q) = %t; want %t", test.vcs, test.url, secure, test.secure)
		}
	}
}

func TestIsSecureGitAllowProtocol(t *testing.T) {
	tests := []struct {
		vcs    *Cmd
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
		secure := test.vcs.IsSecure(test.url)
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
		{
			imports: []metaImport{
				{Prefix: "myitcv.io/blah2", VCS: "mod", RepoRoot: "https://raw.githubusercontent.com/myitcv/pubx/master"},
				{Prefix: "myitcv.io", VCS: "git", RepoRoot: "https://github.com/myitcv/x"},
			},
			path: "myitcv.io/blah2/foo",
			mi:   metaImport{Prefix: "myitcv.io/blah2", VCS: "mod", RepoRoot: "https://raw.githubusercontent.com/myitcv/pubx/master"},
		},
		{
			imports: []metaImport{
				{Prefix: "myitcv.io/blah2", VCS: "mod", RepoRoot: "https://raw.githubusercontent.com/myitcv/pubx/master"},
				{Prefix: "myitcv.io", VCS: "git", RepoRoot: "https://github.com/myitcv/x"},
			},
			path: "myitcv.io/other",
			mi:   metaImport{Prefix: "myitcv.io", VCS: "git", RepoRoot: "https://github.com/myitcv/x"},
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

func TestValidateRepoRoot(t *testing.T) {
	tests := []struct {
		root string
		ok   bool
	}{
		{
			root: "",
			ok:   false,
		},
		{
			root: "http://",
			ok:   true,
		},
		{
			root: "git+ssh://",
			ok:   true,
		},
		{
			root: "http#://",
			ok:   false,
		},
		{
			root: "-config",
			ok:   false,
		},
		{
			root: "-config://",
			ok:   false,
		},
	}

	for _, test := range tests {
		err := validateRepoRoot(test.root)
		ok := err == nil
		if ok != test.ok {
			want := "error"
			if test.ok {
				want = "nil"
			}
			t.Errorf("validateRepoRoot(%q) = %q, want %s", test.root, err, want)
		}
	}
}

var govcsTests = []struct {
	govcs string
	path  string
	vcs   string
	ok    bool
}{
	{"private:all", "is-public.com/foo", "zzz", false},
	{"private:all", "is-private.com/foo", "zzz", true},
	{"public:all", "is-public.com/foo", "zzz", true},
	{"public:all", "is-private.com/foo", "zzz", false},
	{"public:all,private:none", "is-public.com/foo", "zzz", true},
	{"public:all,private:none", "is-private.com/foo", "zzz", false},
	{"*:all", "is-public.com/foo", "zzz", true},
	{"golang.org:git", "golang.org/x/text", "zzz", false},
	{"golang.org:git", "golang.org/x/text", "git", true},
	{"golang.org:zzz", "golang.org/x/text", "zzz", true},
	{"golang.org:zzz", "golang.org/x/text", "git", false},
	{"golang.org:zzz", "golang.org/x/text", "zzz", true},
	{"golang.org:zzz", "golang.org/x/text", "git", false},
	{"golang.org:git|hg", "golang.org/x/text", "hg", true},
	{"golang.org:git|hg", "golang.org/x/text", "git", true},
	{"golang.org:git|hg", "golang.org/x/text", "zzz", false},
	{"golang.org:all", "golang.org/x/text", "hg", true},
	{"golang.org:all", "golang.org/x/text", "git", true},
	{"golang.org:all", "golang.org/x/text", "zzz", true},
	{"other.xyz/p:none,golang.org/x:git", "other.xyz/p/x", "git", false},
	{"other.xyz/p:none,golang.org/x:git", "unexpected.com", "git", false},
	{"other.xyz/p:none,golang.org/x:git", "golang.org/x/text", "zzz", false},
	{"other.xyz/p:none,golang.org/x:git", "golang.org/x/text", "git", true},
	{"other.xyz/p:none,golang.org/x:zzz", "golang.org/x/text", "zzz", true},
	{"other.xyz/p:none,golang.org/x:zzz", "golang.org/x/text", "git", false},
	{"other.xyz/p:none,golang.org/x:git|hg", "golang.org/x/text", "hg", true},
	{"other.xyz/p:none,golang.org/x:git|hg", "golang.org/x/text", "git", true},
	{"other.xyz/p:none,golang.org/x:git|hg", "golang.org/x/text", "zzz", false},
	{"other.xyz/p:none,golang.org/x:all", "golang.org/x/text", "hg", true},
	{"other.xyz/p:none,golang.org/x:all", "golang.org/x/text", "git", true},
	{"other.xyz/p:none,golang.org/x:all", "golang.org/x/text", "zzz", true},
	{"other.xyz/p:none,golang.org/x:git", "golang.org/y/text", "zzz", false},
	{"other.xyz/p:none,golang.org/x:git", "golang.org/y/text", "git", false},
	{"other.xyz/p:none,golang.org/x:zzz", "golang.org/y/text", "zzz", false},
	{"other.xyz/p:none,golang.org/x:zzz", "golang.org/y/text", "git", false},
	{"other.xyz/p:none,golang.org/x:git|hg", "golang.org/y/text", "hg", false},
	{"other.xyz/p:none,golang.org/x:git|hg", "golang.org/y/text", "git", false},
	{"other.xyz/p:none,golang.org/x:git|hg", "golang.org/y/text", "zzz", false},
	{"other.xyz/p:none,golang.org/x:all", "golang.org/y/text", "hg", false},
	{"other.xyz/p:none,golang.org/x:all", "golang.org/y/text", "git", false},
	{"other.xyz/p:none,golang.org/x:all", "golang.org/y/text", "zzz", false},
}

func TestGOVCS(t *testing.T) {
	for _, tt := range govcsTests {
		cfg, err := parseGOVCS(tt.govcs)
		if err != nil {
			t.Errorf("parseGOVCS(%q): %v", tt.govcs, err)
			continue
		}
		private := strings.HasPrefix(tt.path, "is-private")
		ok := cfg.allow(tt.path, private, tt.vcs)
		if ok != tt.ok {
			t.Errorf("parseGOVCS(%q).allow(%q, %v, %q) = %v, want %v",
				tt.govcs, tt.path, private, tt.vcs, ok, tt.ok)
		}
	}
}

var govcsErrors = []struct {
	s   string
	err string
}{
	{`,`, `empty entry in GOVCS`},
	{`,x`, `empty entry in GOVCS`},
	{`x,`, `malformed entry in GOVCS (missing colon): "x"`},
	{`x:y,`, `empty entry in GOVCS`},
	{`x`, `malformed entry in GOVCS (missing colon): "x"`},
	{`x:`, `empty VCS list in GOVCS: "x:"`},
	{`x:|`, `empty VCS name in GOVCS: "x:|"`},
	{`x:y|`, `empty VCS name in GOVCS: "x:y|"`},
	{`x:|y`, `empty VCS name in GOVCS: "x:|y"`},
	{`x:y,z:`, `empty VCS list in GOVCS: "z:"`},
	{`x:y,z:|`, `empty VCS name in GOVCS: "z:|"`},
	{`x:y,z:|w`, `empty VCS name in GOVCS: "z:|w"`},
	{`x:y,z:w|`, `empty VCS name in GOVCS: "z:w|"`},
	{`x:y,z:w||v`, `empty VCS name in GOVCS: "z:w||v"`},
	{`x:y,x:z`, `unreachable pattern in GOVCS: "x:z" after "x:y"`},
}

func TestGOVCSErrors(t *testing.T) {
	for _, tt := range govcsErrors {
		_, err := parseGOVCS(tt.s)
		if err == nil || !strings.Contains(err.Error(), tt.err) {
			t.Errorf("parseGOVCS(%s): err=%v, want %v", tt.s, err, tt.err)
		}
	}
}
