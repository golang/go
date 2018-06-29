// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modfetch

import (
	"archive/zip"
	"internal/testenv"
	"io"
	"io/ioutil"
	"log"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	"cmd/go/internal/modfetch/codehost"
)

func init() {
	isTest = true
}

func TestMain(m *testing.M) {
	os.Exit(testMain(m))
}

func testMain(m *testing.M) int {
	dir, err := ioutil.TempDir("", "gitrepo-test-")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(dir)

	codehost.WorkRoot = dir
	return m.Run()
}

var codeRepoTests = []struct {
	path     string
	lookerr  string
	mpath    string
	rev      string
	err      string
	version  string
	name     string
	short    string
	time     time.Time
	gomod    string
	gomoderr string
	zip      []string
	ziperr   string
}{
	{
		path:    "github.com/rsc/vgotest1",
		rev:     "v0.0.0",
		version: "v0.0.0",
		name:    "80d85c5d4d17598a0e9055e7c175a32b415d6128",
		short:   "80d85c5d4d17",
		time:    time.Date(2018, 2, 19, 23, 10, 6, 0, time.UTC),
		zip: []string{
			"LICENSE",
			"README.md",
			"pkg/p.go",
		},
	},
	{
		path:    "github.com/rsc/vgotest1",
		rev:     "v1.0.0",
		version: "v1.0.0",
		name:    "80d85c5d4d17598a0e9055e7c175a32b415d6128",
		short:   "80d85c5d4d17",
		time:    time.Date(2018, 2, 19, 23, 10, 6, 0, time.UTC),
		zip: []string{
			"LICENSE",
			"README.md",
			"pkg/p.go",
		},
	},
	{
		path:    "github.com/rsc/vgotest1/v2",
		rev:     "v2.0.0",
		version: "v2.0.0",
		name:    "80d85c5d4d17598a0e9055e7c175a32b415d6128",
		short:   "80d85c5d4d17",
		time:    time.Date(2018, 2, 19, 23, 10, 6, 0, time.UTC),
		ziperr:  "missing go.mod",
	},
	{
		path:    "github.com/rsc/vgotest1",
		rev:     "80d85",
		version: "v0.0.0-20180219231006-80d85c5d4d17",
		name:    "80d85c5d4d17598a0e9055e7c175a32b415d6128",
		short:   "80d85c5d4d17",
		time:    time.Date(2018, 2, 19, 23, 10, 6, 0, time.UTC),
		zip: []string{
			"LICENSE",
			"README.md",
			"pkg/p.go",
		},
	},
	{
		path:    "github.com/rsc/vgotest1",
		rev:     "mytag",
		version: "v0.0.0-20180219231006-80d85c5d4d17",
		name:    "80d85c5d4d17598a0e9055e7c175a32b415d6128",
		short:   "80d85c5d4d17",
		time:    time.Date(2018, 2, 19, 23, 10, 6, 0, time.UTC),
		zip: []string{
			"LICENSE",
			"README.md",
			"pkg/p.go",
		},
	},
	{
		path:     "github.com/rsc/vgotest1/v2",
		rev:      "80d85",
		version:  "v2.0.0-20180219231006-80d85c5d4d17",
		name:     "80d85c5d4d17598a0e9055e7c175a32b415d6128",
		short:    "80d85c5d4d17",
		time:     time.Date(2018, 2, 19, 23, 10, 6, 0, time.UTC),
		gomoderr: "missing go.mod",
		ziperr:   "missing go.mod",
	},
	{
		path:    "github.com/rsc/vgotest1/v54321",
		rev:     "80d85",
		version: "v54321.0.0-20180219231006-80d85c5d4d17",
		name:    "80d85c5d4d17598a0e9055e7c175a32b415d6128",
		short:   "80d85c5d4d17",
		time:    time.Date(2018, 2, 19, 23, 10, 6, 0, time.UTC),
		ziperr:  "missing go.mod",
	},
	{
		path: "github.com/rsc/vgotest1/submod",
		rev:  "v1.0.0",
		err:  "unknown revision \"submod/v1.0.0\"",
	},
	{
		path: "github.com/rsc/vgotest1/submod",
		rev:  "v1.0.3",
		err:  "unknown revision \"submod/v1.0.3\"",
	},
	{
		path:    "github.com/rsc/vgotest1/submod",
		rev:     "v1.0.4",
		version: "v1.0.4",
		name:    "8afe2b2efed96e0880ecd2a69b98a53b8c2738b6",
		short:   "8afe2b2efed9",
		time:    time.Date(2018, 2, 19, 23, 12, 7, 0, time.UTC),
		gomod:   "module \"github.com/vgotest1/submod\" // submod/go.mod\n",
		zip: []string{
			"go.mod",
			"pkg/p.go",
			"LICENSE",
		},
	},
	{
		path:    "github.com/rsc/vgotest1",
		rev:     "v1.1.0",
		version: "v1.1.0",
		name:    "b769f2de407a4db81af9c5de0a06016d60d2ea09",
		short:   "b769f2de407a",
		time:    time.Date(2018, 2, 19, 23, 13, 36, 0, time.UTC),
		gomod:   "module \"github.com/rsc/vgotest1\" // root go.mod\nrequire \"github.com/rsc/vgotest1/submod\" v1.0.5\n",
		zip: []string{
			"LICENSE",
			"README.md",
			"go.mod",
			"pkg/p.go",
		},
	},
	{
		path:    "github.com/rsc/vgotest1/v2",
		rev:     "v2.0.1",
		version: "v2.0.1",
		name:    "ea65f87c8f52c15ea68f3bdd9925ef17e20d91e9",
		short:   "ea65f87c8f52",
		time:    time.Date(2018, 2, 19, 23, 14, 23, 0, time.UTC),
		gomod:   "module \"github.com/rsc/vgotest1/v2\" // root go.mod\n",
	},
	{
		path:     "github.com/rsc/vgotest1/v2",
		rev:      "v2.0.3",
		version:  "v2.0.3",
		name:     "f18795870fb14388a21ef3ebc1d75911c8694f31",
		short:    "f18795870fb1",
		time:     time.Date(2018, 2, 19, 23, 16, 4, 0, time.UTC),
		gomoderr: "v2/go.mod has non-.../v2 module path",
	},
	{
		path:     "github.com/rsc/vgotest1/v2",
		rev:      "v2.0.4",
		version:  "v2.0.4",
		name:     "1f863feb76bc7029b78b21c5375644838962f88d",
		short:    "1f863feb76bc",
		time:     time.Date(2018, 2, 20, 0, 3, 38, 0, time.UTC),
		gomoderr: "both go.mod and v2/go.mod claim .../v2 module",
	},
	{
		path:    "github.com/rsc/vgotest1/v2",
		rev:     "v2.0.5",
		version: "v2.0.5",
		name:    "2f615117ce481c8efef46e0cc0b4b4dccfac8fea",
		short:   "2f615117ce48",
		time:    time.Date(2018, 2, 20, 0, 3, 59, 0, time.UTC),
		gomod:   "module \"github.com/rsc/vgotest1/v2\" // v2/go.mod\n",
	},
	{
		path:    "go.googlesource.com/scratch",
		rev:     "0f302529858",
		version: "v0.0.0-20180220024720-0f3025298580",
		name:    "0f30252985809011f026b5a2d5cf456e021623da",
		short:   "0f3025298580",
		time:    time.Date(2018, 2, 20, 2, 47, 20, 0, time.UTC),
		gomod:   "//vgo 0.0.4\n\nmodule go.googlesource.com/scratch\n",
	},
	{
		path:    "go.googlesource.com/scratch/rsc",
		rev:     "0f302529858",
		version: "v0.0.0-20180220024720-0f3025298580",
		name:    "0f30252985809011f026b5a2d5cf456e021623da",
		short:   "0f3025298580",
		time:    time.Date(2018, 2, 20, 2, 47, 20, 0, time.UTC),
		gomod:   "",
	},
	{
		path:     "go.googlesource.com/scratch/cbro",
		rev:      "0f302529858",
		version:  "v0.0.0-20180220024720-0f3025298580",
		name:     "0f30252985809011f026b5a2d5cf456e021623da",
		short:    "0f3025298580",
		time:     time.Date(2018, 2, 20, 2, 47, 20, 0, time.UTC),
		gomoderr: "missing go.mod",
	},
	{
		// redirect to github
		path:    "rsc.io/quote",
		rev:     "v1.0.0",
		version: "v1.0.0",
		name:    "f488df80bcdbd3e5bafdc24ad7d1e79e83edd7e6",
		short:   "f488df80bcdb",
		time:    time.Date(2018, 2, 14, 0, 45, 20, 0, time.UTC),
		gomod:   "module \"rsc.io/quote\"\n",
	},
	{
		// redirect to static hosting proxy
		path:    "swtch.com/testmod",
		rev:     "v1.0.0",
		version: "v1.0.0",
		name:    "v1.0.0",
		short:   "v1.0.0",
		time:    time.Date(1972, 7, 18, 12, 34, 56, 0, time.UTC),
		gomod:   "module \"swtch.com/testmod\"\n",
	},
	{
		// redirect to googlesource
		path:    "golang.org/x/text",
		rev:     "4e4a3210bb",
		version: "v0.0.0-20180208041248-4e4a3210bb54",
		name:    "4e4a3210bb54bb31f6ab2cdca2edcc0b50c420c1",
		short:   "4e4a3210bb54",
		time:    time.Date(2018, 2, 8, 4, 12, 48, 0, time.UTC),
	},
	{
		path:    "github.com/pkg/errors",
		rev:     "v0.8.0",
		version: "v0.8.0",
		name:    "645ef00459ed84a119197bfb8d8205042c6df63d",
		short:   "645ef00459ed",
		time:    time.Date(2016, 9, 29, 1, 48, 1, 0, time.UTC),
	},
	{
		// package in subdirectory - custom domain
		path:    "golang.org/x/net/context",
		lookerr: "module root is \"golang.org/x/net\"",
	},
	{
		// package in subdirectory - github
		path:     "github.com/rsc/quote/buggy",
		rev:      "c4d4236f",
		version:  "v0.0.0-20180214154420-c4d4236f9242",
		name:     "c4d4236f92427c64bfbcf1cc3f8142ab18f30b22",
		short:    "c4d4236f9242",
		time:     time.Date(2018, 2, 14, 15, 44, 20, 0, time.UTC),
		gomoderr: "missing go.mod",
	},
	{
		path:    "gopkg.in/yaml.v2",
		rev:     "d670f940",
		version: "v2.0.0-20180109114331-d670f9405373",
		name:    "d670f9405373e636a5a2765eea47fac0c9bc91a4",
		short:   "d670f9405373",
		time:    time.Date(2018, 1, 9, 11, 43, 31, 0, time.UTC),
		gomod:   "//vgo 0.0.4\n\nmodule gopkg.in/yaml.v2\n",
	},
	{
		path:    "gopkg.in/check.v1",
		rev:     "20d25e280405",
		version: "v1.0.0-20161208181325-20d25e280405",
		name:    "20d25e2804050c1cd24a7eea1e7a6447dd0e74ec",
		short:   "20d25e280405",
		time:    time.Date(2016, 12, 8, 18, 13, 25, 0, time.UTC),
		gomod:   "//vgo 0.0.4\n\nmodule gopkg.in/check.v1\n",
	},
	{
		path:    "gopkg.in/yaml.v2",
		rev:     "v2",
		version: "v2.0.0-20180328195020-5420a8b6744d",
		name:    "5420a8b6744d3b0345ab293f6fcba19c978f1183",
		short:   "5420a8b6744d",
		time:    time.Date(2018, 3, 28, 19, 50, 20, 0, time.UTC),
		gomod:   "module \"gopkg.in/yaml.v2\"\n\nrequire (\n\t\"gopkg.in/check.v1\" v0.0.0-20161208181325-20d25e280405\n)\n",
	},
	{
		path:    "vcs-test.golang.org/go/mod/gitrepo1",
		rev:     "master",
		version: "v0.0.0-20180417194322-ede458df7cd0",
		name:    "ede458df7cd0fdca520df19a33158086a8a68e81",
		short:   "ede458df7cd0",
		time:    time.Date(2018, 4, 17, 19, 43, 22, 0, time.UTC),
		gomod:   "//vgo 0.0.4\n\nmodule vcs-test.golang.org/go/mod/gitrepo1\n",
	},
	{
		path:    "gopkg.in/natefinch/lumberjack.v2",
		rev:     "latest",
		version: "v2.0.0-20170531160350-a96e63847dc3",
		name:    "a96e63847dc3c67d17befa69c303767e2f84e54f",
		short:   "a96e63847dc3",
		time:    time.Date(2017, 5, 31, 16, 3, 50, 0, time.UTC),
		gomod:   "//vgo 0.0.4\n\nmodule gopkg.in/natefinch/lumberjack.v2\n",
	},
	{
		path: "gopkg.in/natefinch/lumberjack.v2",
		// This repo has a v2.1 tag.
		// We only allow semver references to tags that are fully qualified, as in v2.1.0.
		// Because we can't record v2.1.0 (the actual tag is v2.1), we record a pseudo-version
		// instead, same as if the tag were any other non-version-looking string.
		// We use a v2 pseudo-version here because of the .v2 in the path, not because
		// of the v2 in the rev.
		rev:     "v2.1", // non-canonical semantic version turns into pseudo-version
		version: "v2.0.0-20170531160350-a96e63847dc3",
		name:    "a96e63847dc3c67d17befa69c303767e2f84e54f",
		short:   "a96e63847dc3",
		time:    time.Date(2017, 5, 31, 16, 3, 50, 0, time.UTC),
		gomod:   "//vgo 0.0.4\n\nmodule gopkg.in/natefinch/lumberjack.v2\n",
	},
}

func TestCodeRepo(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	tmpdir, err := ioutil.TempDir("", "vgo-modfetch-test-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)
	for _, tt := range codeRepoTests {
		t.Run(strings.Replace(tt.path, "/", "_", -1)+"/"+tt.rev, func(t *testing.T) {
			repo, err := Lookup(tt.path)
			if err != nil {
				if tt.lookerr != "" {
					if err.Error() == tt.lookerr {
						return
					}
					t.Errorf("Lookup(%q): %v, want error %q", tt.path, err, tt.lookerr)
				}
				t.Fatalf("Lookup(%q): %v", tt.path, err)
			}
			if tt.mpath == "" {
				tt.mpath = tt.path
			}
			if mpath := repo.ModulePath(); mpath != tt.mpath {
				t.Errorf("repo.ModulePath() = %q, want %q", mpath, tt.mpath)
			}
			info, err := repo.Stat(tt.rev)
			if err != nil {
				if tt.err != "" {
					if !strings.Contains(err.Error(), tt.err) {
						t.Fatalf("repoStat(%q): %v, wanted %q", tt.rev, err, tt.err)
					}
					return
				}
				t.Fatalf("repo.Stat(%q): %v", tt.rev, err)
			}
			if tt.err != "" {
				t.Errorf("repo.Stat(%q): success, wanted error", tt.rev)
			}
			if info.Version != tt.version {
				t.Errorf("info.Version = %q, want %q", info.Version, tt.version)
			}
			if info.Name != tt.name {
				t.Errorf("info.Name = %q, want %q", info.Name, tt.name)
			}
			if info.Short != tt.short {
				t.Errorf("info.Short = %q, want %q", info.Short, tt.short)
			}
			if !info.Time.Equal(tt.time) {
				t.Errorf("info.Time = %v, want %v", info.Time, tt.time)
			}
			if tt.gomod != "" || tt.gomoderr != "" {
				data, err := repo.GoMod(tt.version)
				if err != nil && tt.gomoderr == "" {
					t.Errorf("repo.GoMod(%q): %v", tt.version, err)
				} else if err != nil && tt.gomoderr != "" {
					if err.Error() != tt.gomoderr {
						t.Errorf("repo.GoMod(%q): %v, want %q", tt.version, err, tt.gomoderr)
					}
				} else if tt.gomoderr != "" {
					t.Errorf("repo.GoMod(%q) = %q, want error %q", tt.version, data, tt.gomoderr)
				} else if string(data) != tt.gomod {
					t.Errorf("repo.GoMod(%q) = %q, want %q", tt.version, data, tt.gomod)
				}
			}
			if tt.zip != nil || tt.ziperr != "" {
				zipfile, err := repo.Zip(tt.version, tmpdir)
				if err != nil {
					if tt.ziperr != "" {
						if err.Error() == tt.ziperr {
							return
						}
						t.Fatalf("repo.Zip(%q): %v, want error %q", tt.version, err, tt.ziperr)
					}
					t.Fatalf("repo.Zip(%q): %v", tt.version, err)
				}
				if tt.ziperr != "" {
					t.Errorf("repo.Zip(%q): success, want error %q", tt.version, tt.ziperr)
				}
				prefix := tt.path + "@" + tt.version + "/"
				z, err := zip.OpenReader(zipfile)
				if err != nil {
					t.Fatalf("open zip %s: %v", zipfile, err)
				}
				var names []string
				for _, file := range z.File {
					if !strings.HasPrefix(file.Name, prefix) {
						t.Errorf("zip entry %v does not start with prefix %v", file.Name, prefix)
						continue
					}
					names = append(names, file.Name[len(prefix):])
				}
				z.Close()
				if !reflect.DeepEqual(names, tt.zip) {
					t.Fatalf("zip = %v\nwant %v\n", names, tt.zip)
				}
			}
		})
	}
}

var importTests = []struct {
	path  string
	mpath string
	err   string
}{
	{
		path:  "golang.org/x/net/context",
		mpath: "golang.org/x/net",
	},
	{
		path:  "github.com/rsc/quote/buggy",
		mpath: "github.com/rsc/quote",
	},
	{
		path:  "golang.org/x/net",
		mpath: "golang.org/x/net",
	},
	{
		path:  "github.com/rsc/quote",
		mpath: "github.com/rsc/quote",
	},
	{
		path: "golang.org/x/foo/bar",
		err:  "unknown module golang.org/x/foo/bar: no go-import tags",
	},
}

func TestImport(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	for _, tt := range importTests {
		t.Run(strings.Replace(tt.path, "/", "_", -1), func(t *testing.T) {
			repo, info, err := Import(tt.path, nil)
			if err != nil {
				if tt.err != "" {
					if err.Error() == tt.err {
						return
					}
					t.Errorf("Import(%q): %v, want error %q", tt.path, err, tt.err)
				}
				t.Fatalf("Lookup(%q): %v", tt.path, err)
			}
			if mpath := repo.ModulePath(); mpath != tt.mpath {
				t.Errorf("repo.ModulePath() = %q (%v), want %q", mpath, info.Version, tt.mpath)
			}
		})
	}
}

var codeRepoVersionsTests = []struct {
	path     string
	prefix   string
	versions []string
}{
	// TODO: Why do we allow a prefix here at all?
	{
		path:     "github.com/rsc/vgotest1",
		versions: []string{"v0.0.0", "v0.0.1", "v1.0.0", "v1.0.1", "v1.0.2", "v1.0.3", "v1.1.0"},
	},
	{
		path:     "github.com/rsc/vgotest1",
		prefix:   "v1.0",
		versions: []string{"v1.0.0", "v1.0.1", "v1.0.2", "v1.0.3"},
	},
	{
		path:     "github.com/rsc/vgotest1/v2",
		versions: []string{"v2.0.0", "v2.0.1", "v2.0.2", "v2.0.3", "v2.0.4", "v2.0.5", "v2.0.6"},
	},
	{
		path:     "swtch.com/testmod",
		versions: []string{"v1.0.0", "v1.1.1"},
	},
	{
		path:     "gopkg.in/russross/blackfriday.v2",
		versions: []string{"v2.0.0"},
	},
	{
		path:     "gopkg.in/natefinch/lumberjack.v2",
		versions: []string{},
	},
}

func TestCodeRepoVersions(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	tmpdir, err := ioutil.TempDir("", "vgo-modfetch-test-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)
	for _, tt := range codeRepoVersionsTests {
		t.Run(strings.Replace(tt.path, "/", "_", -1), func(t *testing.T) {
			repo, err := Lookup(tt.path)
			if err != nil {
				t.Fatalf("Lookup(%q): %v", tt.path, err)
			}
			list, err := repo.Versions(tt.prefix)
			if err != nil {
				t.Fatalf("Versions(%q): %v", tt.prefix, err)
			}
			if !reflect.DeepEqual(list, tt.versions) {
				t.Fatalf("Versions(%q):\nhave %v\nwant %v", tt.prefix, list, tt.versions)
			}
		})
	}
}

var latestTests = []struct {
	path    string
	version string
	err     string
}{
	{
		path: "github.com/rsc/empty",
		err:  "no commits",
	},
	{
		path:    "github.com/rsc/vgotest1",
		version: "v0.0.0-20180219223237-a08abb797a67",
	},
	{
		path:    "swtch.com/testmod",
		version: "v1.1.1",
	},
}

func TestLatest(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	tmpdir, err := ioutil.TempDir("", "vgo-modfetch-test-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)
	for _, tt := range latestTests {
		name := strings.Replace(tt.path, "/", "_", -1)
		t.Run(name, func(t *testing.T) {
			repo, err := Lookup(tt.path)
			if err != nil {
				t.Fatalf("Lookup(%q): %v", tt.path, err)
			}
			info, err := repo.Latest()
			if err != nil {
				if tt.err != "" {
					if err.Error() == tt.err {
						return
					}
					t.Fatalf("Latest(): %v, want %q", err, tt.err)
				}
				t.Fatalf("Latest(): %v", err)
			}
			if info.Version != tt.version {
				t.Fatalf("Latest() = %v, want %v", info.Version, tt.version)
			}
		})
	}
}

// fixedTagsRepo is a fake codehost.Repo that returns a fixed list of tags
type fixedTagsRepo struct {
	root string
	tags []string
}

func (ch *fixedTagsRepo) Tags(string) ([]string, error)                  { return ch.tags, nil }
func (ch *fixedTagsRepo) Root() string                                   { return ch.root }
func (ch *fixedTagsRepo) Latest() (*codehost.RevInfo, error)             { panic("not impl") }
func (ch *fixedTagsRepo) ReadFile(string, string, int64) ([]byte, error) { panic("not impl") }
func (ch *fixedTagsRepo) ReadZip(string, string, int64) (io.ReadCloser, string, error) {
	panic("not impl")
}
func (ch *fixedTagsRepo) Stat(string) (*codehost.RevInfo, error) { panic("not impl") }

func TestNonCanonicalSemver(t *testing.T) {
	root := "golang.org/x/issue24476"
	ch := &fixedTagsRepo{
		root: root,
		tags: []string{
			"", "huh?", "1.0.1",
			// what about "version 1 dot dogcow"?
			"v1.🐕.🐄",
			"v1", "v0.1",
			// and one normal one that should pass through
			"v1.0.1",
		},
	}

	cr, err := newCodeRepo(ch, root)
	if err != nil {
		t.Fatal(err)
	}

	v, err := cr.Versions("")
	if err != nil {
		t.Fatal(err)
	}
	if len(v) != 1 || v[0] != "v1.0.1" {
		t.Fatal("unexpected versions returned:", v)
	}
}

var modPathTests = []struct {
	input    []byte
	expected string
}{
	{input: []byte("module \"github.com/rsc/vgotest\""), expected: "github.com/rsc/vgotest"},
	{input: []byte("module github.com/rsc/vgotest"), expected: "github.com/rsc/vgotest"},
	{input: []byte("module  \"github.com/rsc/vgotest\""), expected: "github.com/rsc/vgotest"},
	{input: []byte("module  github.com/rsc/vgotest"), expected: "github.com/rsc/vgotest"},
	{input: []byte("module `github.com/rsc/vgotest`"), expected: "github.com/rsc/vgotest"},
	{input: []byte("module \"github.com/rsc/vgotest/v2\""), expected: "github.com/rsc/vgotest/v2"},
	{input: []byte("module github.com/rsc/vgotest/v2"), expected: "github.com/rsc/vgotest/v2"},
	{input: []byte("module \"gopkg.in/yaml.v2\""), expected: "gopkg.in/yaml.v2"},
	{input: []byte("module gopkg.in/yaml.v2"), expected: "gopkg.in/yaml.v2"},
	{input: []byte("module \"gopkg.in/check.v1\"\n"), expected: "gopkg.in/check.v1"},
	{input: []byte("module \"gopkg.in/check.v1\n\""), expected: ""},
	{input: []byte("module gopkg.in/check.v1\n"), expected: "gopkg.in/check.v1"},
	{input: []byte("module \"gopkg.in/check.v1\"\r\n"), expected: "gopkg.in/check.v1"},
	{input: []byte("module gopkg.in/check.v1\r\n"), expected: "gopkg.in/check.v1"},
	{input: []byte("module \"gopkg.in/check.v1\"\n\n"), expected: "gopkg.in/check.v1"},
	{input: []byte("module gopkg.in/check.v1\n\n"), expected: "gopkg.in/check.v1"},
	{input: []byte("module \n\"gopkg.in/check.v1\"\n\n"), expected: ""},
	{input: []byte("module \ngopkg.in/check.v1\n\n"), expected: ""},
	{input: []byte("module \"gopkg.in/check.v1\"asd"), expected: ""},
}

func TestModPath(t *testing.T) {
	for _, test := range modPathTests {
		t.Run(string(test.input), func(t *testing.T) {
			result := modPath(test.input)
			if result != test.expected {
				t.Fatalf("modPath(%s): %s, want %s", string(test.input), result, test.expected)
			}
		})
	}
}
