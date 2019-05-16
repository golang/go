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

	"cmd/go/internal/cfg"
	"cmd/go/internal/modfetch/codehost"
)

func TestMain(m *testing.M) {
	os.Exit(testMain(m))
}

func testMain(m *testing.M) int {
	cfg.GOPROXY = "direct"

	// The sum database is populated using a released version of the go command,
	// but this test may include fixes for additional modules that previously
	// could not be fetched. Since this test isn't executing any of the resolved
	// code, bypass the sum database.
	cfg.GOSUMDB = "off"

	dir, err := ioutil.TempDir("", "gitrepo-test-")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(dir)

	codehost.WorkRoot = dir
	return m.Run()
}

const (
	vgotest1git = "github.com/rsc/vgotest1"
	vgotest1hg  = "vcs-test.golang.org/hg/vgotest1.hg"
)

var altVgotests = []string{
	vgotest1hg,
}

type codeRepoTest struct {
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
}

var codeRepoTests = []codeRepoTest{
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
		name:    "45f53230a74ad275c7127e117ac46914c8126160",
		short:   "45f53230a74a",
		time:    time.Date(2018, 7, 19, 1, 21, 27, 0, time.UTC),
		ziperr:  "missing github.com/rsc/vgotest1/go.mod and .../v2/go.mod at revision v2.0.0",
	},
	{
		path:    "github.com/rsc/vgotest1",
		rev:     "80d85c5",
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
		path:    "github.com/rsc/vgotest1",
		rev:     "mytag",
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
		path:     "github.com/rsc/vgotest1/v2",
		rev:      "45f53230a",
		version:  "v2.0.0",
		name:     "45f53230a74ad275c7127e117ac46914c8126160",
		short:    "45f53230a74a",
		time:     time.Date(2018, 7, 19, 1, 21, 27, 0, time.UTC),
		gomoderr: "missing github.com/rsc/vgotest1/go.mod and .../v2/go.mod at revision v2.0.0",
		ziperr:   "missing github.com/rsc/vgotest1/go.mod and .../v2/go.mod at revision v2.0.0",
	},
	{
		path:    "github.com/rsc/vgotest1/v54321",
		rev:     "80d85c5",
		version: "v54321.0.0-20180219231006-80d85c5d4d17",
		name:    "80d85c5d4d17598a0e9055e7c175a32b415d6128",
		short:   "80d85c5d4d17",
		time:    time.Date(2018, 2, 19, 23, 10, 6, 0, time.UTC),
		ziperr:  "missing github.com/rsc/vgotest1/go.mod and .../v54321/go.mod at revision 80d85c5d4d17",
	},
	{
		path: "github.com/rsc/vgotest1/submod",
		rev:  "v1.0.0",
		err:  "unknown revision submod/v1.0.0",
	},
	{
		path: "github.com/rsc/vgotest1/submod",
		rev:  "v1.0.3",
		err:  "unknown revision submod/v1.0.3",
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
		gomoderr: "github.com/rsc/vgotest1/v2/go.mod has non-.../v2 module path \"github.com/rsc/vgotest\" at revision v2.0.3",
	},
	{
		path:     "github.com/rsc/vgotest1/v2",
		rev:      "v2.0.4",
		version:  "v2.0.4",
		name:     "1f863feb76bc7029b78b21c5375644838962f88d",
		short:    "1f863feb76bc",
		time:     time.Date(2018, 2, 20, 0, 3, 38, 0, time.UTC),
		gomoderr: "github.com/rsc/vgotest1/go.mod and .../v2/go.mod both have .../v2 module paths at revision v2.0.4",
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
		// NO name or short - we intentionally ignore those in the proxy protocol
		time:  time.Date(1972, 7, 18, 12, 34, 56, 0, time.UTC),
		gomod: "module \"swtch.com/testmod\"\n",
	},
	{
		// redirect to googlesource
		path:    "golang.org/x/text",
		rev:     "4e4a3210bb",
		version: "v0.3.1-0.20180208041248-4e4a3210bb54",
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
		// In general we can't reject these definitively in Lookup,
		// but gopkg.in is special.
		path:    "gopkg.in/yaml.v2/abc",
		lookerr: "invalid module path \"gopkg.in/yaml.v2/abc\"",
	},
	{
		// package in subdirectory - github
		// Because it's a package, Stat should fail entirely.
		path: "github.com/rsc/quote/buggy",
		rev:  "c4d4236f",
		err:  "missing github.com/rsc/quote/buggy/go.mod at revision c4d4236f9242",
	},
	{
		path:    "gopkg.in/yaml.v2",
		rev:     "d670f940",
		version: "v2.0.0",
		name:    "d670f9405373e636a5a2765eea47fac0c9bc91a4",
		short:   "d670f9405373",
		time:    time.Date(2018, 1, 9, 11, 43, 31, 0, time.UTC),
		gomod:   "module gopkg.in/yaml.v2\n",
	},
	{
		path:    "gopkg.in/check.v1",
		rev:     "20d25e280405",
		version: "v1.0.0-20161208181325-20d25e280405",
		name:    "20d25e2804050c1cd24a7eea1e7a6447dd0e74ec",
		short:   "20d25e280405",
		time:    time.Date(2016, 12, 8, 18, 13, 25, 0, time.UTC),
		gomod:   "module gopkg.in/check.v1\n",
	},
	{
		path:    "gopkg.in/yaml.v2",
		rev:     "v2",
		version: "v2.2.3-0.20190319135612-7b8349ac747c",
		name:    "7b8349ac747c6a24702b762d2c4fd9266cf4f1d6",
		short:   "7b8349ac747c",
		time:    time.Date(2019, 03, 19, 13, 56, 12, 0, time.UTC),
		gomod:   "module \"gopkg.in/yaml.v2\"\n\nrequire (\n\t\"gopkg.in/check.v1\" v0.0.0-20161208181325-20d25e280405\n)\n",
	},
	{
		path:    "vcs-test.golang.org/go/mod/gitrepo1",
		rev:     "master",
		version: "v1.2.4-annotated",
		name:    "ede458df7cd0fdca520df19a33158086a8a68e81",
		short:   "ede458df7cd0",
		time:    time.Date(2018, 4, 17, 19, 43, 22, 0, time.UTC),
		gomod:   "module vcs-test.golang.org/go/mod/gitrepo1\n",
	},
	{
		path:    "gopkg.in/natefinch/lumberjack.v2",
		rev:     "latest",
		version: "v2.0.0-20170531160350-a96e63847dc3",
		name:    "a96e63847dc3c67d17befa69c303767e2f84e54f",
		short:   "a96e63847dc3",
		time:    time.Date(2017, 5, 31, 16, 3, 50, 0, time.UTC),
		gomod:   "module gopkg.in/natefinch/lumberjack.v2\n",
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
		gomod:   "module gopkg.in/natefinch/lumberjack.v2\n",
	},
	{
		path:    "vcs-test.golang.org/go/v2module/v2",
		rev:     "v2.0.0",
		version: "v2.0.0",
		name:    "203b91c896acd173aa719e4cdcb7d463c4b090fa",
		short:   "203b91c896ac",
		time:    time.Date(2019, 4, 3, 15, 52, 15, 0, time.UTC),
		gomod:   "module vcs-test.golang.org/go/v2module/v2\n\ngo 1.12\n",
	},
}

func TestCodeRepo(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	tmpdir, err := ioutil.TempDir("", "modfetch-test-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	t.Run("parallel", func(t *testing.T) {
		for _, tt := range codeRepoTests {
			f := func(tt codeRepoTest) func(t *testing.T) {
				return func(t *testing.T) {
					t.Parallel()

					repo, err := Lookup("direct", tt.path)
					if tt.lookerr != "" {
						if err != nil && err.Error() == tt.lookerr {
							return
						}
						t.Errorf("Lookup(%q): %v, want error %q", tt.path, err, tt.lookerr)
					}
					if err != nil {
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
						f, err := ioutil.TempFile(tmpdir, tt.version+".zip.")
						if err != nil {
							t.Fatalf("ioutil.TempFile: %v", err)
						}
						zipfile := f.Name()
						err = repo.Zip(f, tt.version)
						f.Close()
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
				}
			}
			t.Run(strings.ReplaceAll(tt.path, "/", "_")+"/"+tt.rev, f(tt))
			if strings.HasPrefix(tt.path, vgotest1git) {
				for _, alt := range altVgotests {
					// Note: Communicating with f through tt; should be cleaned up.
					old := tt
					tt.path = alt + strings.TrimPrefix(tt.path, vgotest1git)
					if strings.HasPrefix(tt.mpath, vgotest1git) {
						tt.mpath = alt + strings.TrimPrefix(tt.mpath, vgotest1git)
					}
					var m map[string]string
					if alt == vgotest1hg {
						m = hgmap
					}
					tt.version = remap(tt.version, m)
					tt.name = remap(tt.name, m)
					tt.short = remap(tt.short, m)
					tt.rev = remap(tt.rev, m)
					tt.gomoderr = remap(tt.gomoderr, m)
					tt.ziperr = remap(tt.ziperr, m)
					t.Run(strings.ReplaceAll(tt.path, "/", "_")+"/"+tt.rev, f(tt))
					tt = old
				}
			}
		}
	})
}

var hgmap = map[string]string{
	"github.com/rsc/vgotest1/":                 "vcs-test.golang.org/hg/vgotest1.hg/",
	"f18795870fb14388a21ef3ebc1d75911c8694f31": "a9ad6d1d14eb544f459f446210c7eb3b009807c6",
	"ea65f87c8f52c15ea68f3bdd9925ef17e20d91e9": "f1fc0f22021b638d073d31c752847e7bf385def7",
	"b769f2de407a4db81af9c5de0a06016d60d2ea09": "92c7eb888b4fac17f1c6bd2e1060a1b881a3b832",
	"8afe2b2efed96e0880ecd2a69b98a53b8c2738b6": "4e58084d459ae7e79c8c2264d0e8e9a92eb5cd44",
	"2f615117ce481c8efef46e0cc0b4b4dccfac8fea": "879ea98f7743c8eff54f59a918f3a24123d1cf46",
	"80d85c5d4d17598a0e9055e7c175a32b415d6128": "e125018e286a4b09061079a81e7b537070b7ff71",
	"1f863feb76bc7029b78b21c5375644838962f88d": "bf63880162304a9337477f3858f5b7e255c75459",
	"45f53230a74ad275c7127e117ac46914c8126160": "814fce58e83abd5bf2a13892e0b0e1198abefcd4",
}

func remap(name string, m map[string]string) string {
	if m[name] != "" {
		return m[name]
	}
	if codehost.AllHex(name) {
		for k, v := range m {
			if strings.HasPrefix(k, name) {
				return v[:len(name)]
			}
		}
	}
	for k, v := range m {
		name = strings.ReplaceAll(name, k, v)
		if codehost.AllHex(k) {
			name = strings.ReplaceAll(name, k[:12], v[:12])
		}
	}
	return name
}

var codeRepoVersionsTests = []struct {
	path     string
	prefix   string
	versions []string
}{
	{
		path:     "github.com/rsc/vgotest1",
		versions: []string{"v0.0.0", "v0.0.1", "v1.0.0", "v1.0.1", "v1.0.2", "v1.0.3", "v1.1.0", "v2.0.0+incompatible"},
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
		versions: []string{"v2.0.0", "v2.0.1"},
	},
	{
		path:     "gopkg.in/natefinch/lumberjack.v2",
		versions: []string{"v2.0.0"},
	},
}

func TestCodeRepoVersions(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	tmpdir, err := ioutil.TempDir("", "vgo-modfetch-test-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	t.Run("parallel", func(t *testing.T) {
		for _, tt := range codeRepoVersionsTests {
			t.Run(strings.ReplaceAll(tt.path, "/", "_"), func(t *testing.T) {
				tt := tt
				t.Parallel()

				repo, err := Lookup("direct", tt.path)
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
	})
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
		path: "github.com/rsc/vgotest1/subdir",
		err:  "missing github.com/rsc/vgotest1/subdir/go.mod at revision a08abb797a67",
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

	t.Run("parallel", func(t *testing.T) {
		for _, tt := range latestTests {
			name := strings.ReplaceAll(tt.path, "/", "_")
			t.Run(name, func(t *testing.T) {
				tt := tt
				t.Parallel()

				repo, err := Lookup("direct", tt.path)
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
				if tt.err != "" {
					t.Fatalf("Latest() = %v, want error %q", info.Version, tt.err)
				}
				if info.Version != tt.version {
					t.Fatalf("Latest() = %v, want %v", info.Version, tt.version)
				}
			})
		}
	})
}

// fixedTagsRepo is a fake codehost.Repo that returns a fixed list of tags
type fixedTagsRepo struct {
	tags []string
}

func (ch *fixedTagsRepo) Tags(string) ([]string, error)                  { return ch.tags, nil }
func (ch *fixedTagsRepo) Latest() (*codehost.RevInfo, error)             { panic("not impl") }
func (ch *fixedTagsRepo) ReadFile(string, string, int64) ([]byte, error) { panic("not impl") }
func (ch *fixedTagsRepo) ReadFileRevs([]string, string, int64) (map[string]*codehost.FileRev, error) {
	panic("not impl")
}
func (ch *fixedTagsRepo) ReadZip(string, string, int64) (io.ReadCloser, string, error) {
	panic("not impl")
}
func (ch *fixedTagsRepo) RecentTag(string, string) (string, error) {
	panic("not impl")
}
func (ch *fixedTagsRepo) Stat(string) (*codehost.RevInfo, error) { panic("not impl") }

func TestNonCanonicalSemver(t *testing.T) {
	root := "golang.org/x/issue24476"
	ch := &fixedTagsRepo{
		tags: []string{
			"", "huh?", "1.0.1",
			// what about "version 1 dot dogcow"?
			"v1.🐕.🐄",
			"v1", "v0.1",
			// and one normal one that should pass through
			"v1.0.1",
		},
	}

	cr, err := newCodeRepo(ch, root, root)
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
