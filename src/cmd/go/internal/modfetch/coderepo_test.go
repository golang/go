// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modfetch

import (
	"archive/zip"
	"crypto/sha256"
	"encoding/hex"
	"hash"
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

	"golang.org/x/mod/sumdb/dirhash"
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

	cfg.GOMODCACHE = dir
	return m.Run()
}

const (
	vgotest1git = "github.com/rsc/vgotest1"
	vgotest1hg  = "vcs-test.golang.org/hg/vgotest1.hg"
)

var altVgotests = map[string]string{
	"hg": vgotest1hg,
}

type codeRepoTest struct {
	vcs         string
	path        string
	lookErr     string
	mpath       string
	rev         string
	err         string
	version     string
	name        string
	short       string
	time        time.Time
	gomod       string
	gomodErr    string
	zip         []string
	zipErr      string
	zipSum      string
	zipFileHash string
}

var codeRepoTests = []codeRepoTest{
	{
		vcs:     "git",
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
		zipSum:      "h1:zVEjciLdlk/TPWCOyZo7k24T+tOKRQC+u8MKq/xS80I=",
		zipFileHash: "738a00ddbfe8c329dce6b48e1f23c8e22a92db50f3cfb2653caa0d62676bc09c",
	},
	{
		vcs:     "git",
		path:    "github.com/rsc/vgotest1",
		rev:     "v0.0.0-20180219231006-80d85c5d4d17",
		version: "v0.0.0-20180219231006-80d85c5d4d17",
		name:    "80d85c5d4d17598a0e9055e7c175a32b415d6128",
		short:   "80d85c5d4d17",
		time:    time.Date(2018, 2, 19, 23, 10, 6, 0, time.UTC),
		zip: []string{
			"LICENSE",
			"README.md",
			"pkg/p.go",
		},
		zipSum:      "h1:nOznk2xKsLGkTnXe0q9t1Ewt9jxK+oadtafSUqHM3Ec=",
		zipFileHash: "bacb08f391e29d2eaaef8281b5c129ee6d890e608ee65877e0003c0181a766c8",
	},
	{
		vcs:  "git",
		path: "github.com/rsc/vgotest1",
		rev:  "v0.0.1-0.20180219231006-80d85c5d4d17",
		err:  `github.com/rsc/vgotest1@v0.0.1-0.20180219231006-80d85c5d4d17: invalid pseudo-version: tag (v0.0.0) found on revision 80d85c5d4d17 is already canonical, so should not be replaced with a pseudo-version derived from that tag`,
	},
	{
		vcs:     "git",
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
		zipSum:      "h1:e040hOoWGeuJLawDjK9DW6med+cz9FxMFYDMOVG8ctQ=",
		zipFileHash: "74caab65cfbea427c341fa815f3bb0378681d8f0e3cf62a7f207014263ec7be3",
	},
	{
		vcs:     "git",
		path:    "github.com/rsc/vgotest1/v2",
		rev:     "v2.0.0",
		version: "v2.0.0",
		name:    "45f53230a74ad275c7127e117ac46914c8126160",
		short:   "45f53230a74a",
		time:    time.Date(2018, 7, 19, 1, 21, 27, 0, time.UTC),
		err:     "missing github.com/rsc/vgotest1/go.mod and .../v2/go.mod at revision v2.0.0",
	},
	{
		vcs:     "git",
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
		zipSum:      "h1:e040hOoWGeuJLawDjK9DW6med+cz9FxMFYDMOVG8ctQ=",
		zipFileHash: "74caab65cfbea427c341fa815f3bb0378681d8f0e3cf62a7f207014263ec7be3",
	},
	{
		vcs:     "git",
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
		vcs:     "git",
		path:    "github.com/rsc/vgotest1/v2",
		rev:     "45f53230a",
		version: "v2.0.0",
		name:    "45f53230a74ad275c7127e117ac46914c8126160",
		short:   "45f53230a74a",
		time:    time.Date(2018, 7, 19, 1, 21, 27, 0, time.UTC),
		err:     "missing github.com/rsc/vgotest1/go.mod and .../v2/go.mod at revision v2.0.0",
	},
	{
		vcs:     "git",
		path:    "github.com/rsc/vgotest1/v54321",
		rev:     "80d85c5",
		version: "v54321.0.0-20180219231006-80d85c5d4d17",
		name:    "80d85c5d4d17598a0e9055e7c175a32b415d6128",
		short:   "80d85c5d4d17",
		time:    time.Date(2018, 2, 19, 23, 10, 6, 0, time.UTC),
		err:     "missing github.com/rsc/vgotest1/go.mod and .../v54321/go.mod at revision 80d85c5d4d17",
	},
	{
		vcs:  "git",
		path: "github.com/rsc/vgotest1/submod",
		rev:  "v1.0.0",
		err:  "unknown revision submod/v1.0.0",
	},
	{
		vcs:  "git",
		path: "github.com/rsc/vgotest1/submod",
		rev:  "v1.0.3",
		err:  "unknown revision submod/v1.0.3",
	},
	{
		vcs:     "git",
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
		zipSum:      "h1:iMsJ/9uQsk6MnZNnJK311f11QiSlmN92Q2aSjCywuJY=",
		zipFileHash: "95801bfa69c5197ae809af512946d22f22850068527cd78100ae3f176bc8043b",
	},
	{
		vcs:     "git",
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
		zipSum:      "h1:M69k7q+8bQ+QUpHov45Z/NoR8rj3DsQJUnXLWvf01+Q=",
		zipFileHash: "58af45fb248d320ea471f568e006379e2b8d71d6d1663f9b19b2e00fd9ac9265",
	},
	{
		vcs:         "git",
		path:        "github.com/rsc/vgotest1/v2",
		rev:         "v2.0.1",
		version:     "v2.0.1",
		name:        "ea65f87c8f52c15ea68f3bdd9925ef17e20d91e9",
		short:       "ea65f87c8f52",
		time:        time.Date(2018, 2, 19, 23, 14, 23, 0, time.UTC),
		gomod:       "module \"github.com/rsc/vgotest1/v2\" // root go.mod\n",
		zipSum:      "h1:QmgYy/zt+uoWhDpcsgrSVzYFvKtBEjl5zT/FRz9GTzA=",
		zipFileHash: "1aedf1546d322a0121879ddfd6d0e8bfbd916d2cafbeb538ddb440e04b04b9ef",
	},
	{
		vcs:     "git",
		path:    "github.com/rsc/vgotest1/v2",
		rev:     "v2.0.3",
		version: "v2.0.3",
		name:    "f18795870fb14388a21ef3ebc1d75911c8694f31",
		short:   "f18795870fb1",
		time:    time.Date(2018, 2, 19, 23, 16, 4, 0, time.UTC),
		err:     "github.com/rsc/vgotest1/v2/go.mod has non-.../v2 module path \"github.com/rsc/vgotest\" at revision v2.0.3",
	},
	{
		vcs:     "git",
		path:    "github.com/rsc/vgotest1/v2",
		rev:     "v2.0.4",
		version: "v2.0.4",
		name:    "1f863feb76bc7029b78b21c5375644838962f88d",
		short:   "1f863feb76bc",
		time:    time.Date(2018, 2, 20, 0, 3, 38, 0, time.UTC),
		err:     "github.com/rsc/vgotest1/go.mod and .../v2/go.mod both have .../v2 module paths at revision v2.0.4",
	},
	{
		vcs:         "git",
		path:        "github.com/rsc/vgotest1/v2",
		rev:         "v2.0.5",
		version:     "v2.0.5",
		name:        "2f615117ce481c8efef46e0cc0b4b4dccfac8fea",
		short:       "2f615117ce48",
		time:        time.Date(2018, 2, 20, 0, 3, 59, 0, time.UTC),
		gomod:       "module \"github.com/rsc/vgotest1/v2\" // v2/go.mod\n",
		zipSum:      "h1:RIEb9q1SUSEQOzMn0zfl/LQxGFWlhWEAdeEguf1MLGU=",
		zipFileHash: "7d92c2c328c5e9b0694101353705d5843746ec1d93a1e986d0da54c8a14dfe6d",
	},
	{
		// redirect to github
		vcs:         "git",
		path:        "rsc.io/quote",
		rev:         "v1.0.0",
		version:     "v1.0.0",
		name:        "f488df80bcdbd3e5bafdc24ad7d1e79e83edd7e6",
		short:       "f488df80bcdb",
		time:        time.Date(2018, 2, 14, 0, 45, 20, 0, time.UTC),
		gomod:       "module \"rsc.io/quote\"\n",
		zipSum:      "h1:haUSojyo3j2M9g7CEUFG8Na09dtn7QKxvPGaPVQdGwM=",
		zipFileHash: "5c08ba2c09a364f93704aaa780e7504346102c6ef4fe1333a11f09904a732078",
	},
	{
		// redirect to static hosting proxy
		vcs:     "mod",
		path:    "swtch.com/testmod",
		rev:     "v1.0.0",
		version: "v1.0.0",
		// NO name or short - we intentionally ignore those in the proxy protocol
		time:  time.Date(1972, 7, 18, 12, 34, 56, 0, time.UTC),
		gomod: "module \"swtch.com/testmod\"\n",
	},
	{
		// redirect to googlesource
		vcs:         "git",
		path:        "golang.org/x/text",
		rev:         "4e4a3210bb",
		version:     "v0.3.1-0.20180208041248-4e4a3210bb54",
		name:        "4e4a3210bb54bb31f6ab2cdca2edcc0b50c420c1",
		short:       "4e4a3210bb54",
		time:        time.Date(2018, 2, 8, 4, 12, 48, 0, time.UTC),
		zipSum:      "h1:Yxu6pHX9X2RECiuw/Q5/4uvajuaowck8zOFKXgbfNBk=",
		zipFileHash: "ac2c165a5c10aa5a7545dea60a08e019270b982fa6c8bdcb5943931de64922fe",
	},
	{
		vcs:         "git",
		path:        "github.com/pkg/errors",
		rev:         "v0.8.0",
		version:     "v0.8.0",
		name:        "645ef00459ed84a119197bfb8d8205042c6df63d",
		short:       "645ef00459ed",
		time:        time.Date(2016, 9, 29, 1, 48, 1, 0, time.UTC),
		zipSum:      "h1:WdK/asTD0HN+q6hsWO3/vpuAkAr+tw6aNJNDFFf0+qw=",
		zipFileHash: "e4fa69ba057356614edbc1da881a7d3ebb688505be49f65965686bcb859e2fae",
	},
	{
		// package in subdirectory - custom domain
		// In general we can't reject these definitively in Lookup,
		// but gopkg.in is special.
		vcs:     "git",
		path:    "gopkg.in/yaml.v2/abc",
		lookErr: "invalid module path \"gopkg.in/yaml.v2/abc\"",
	},
	{
		// package in subdirectory - github
		// Because it's a package, Stat should fail entirely.
		vcs:  "git",
		path: "github.com/rsc/quote/buggy",
		rev:  "c4d4236f",
		err:  "missing github.com/rsc/quote/buggy/go.mod at revision c4d4236f9242",
	},
	{
		vcs:         "git",
		path:        "gopkg.in/yaml.v2",
		rev:         "d670f940",
		version:     "v2.0.0",
		name:        "d670f9405373e636a5a2765eea47fac0c9bc91a4",
		short:       "d670f9405373",
		time:        time.Date(2018, 1, 9, 11, 43, 31, 0, time.UTC),
		gomod:       "module gopkg.in/yaml.v2\n",
		zipSum:      "h1:uUkhRGrsEyx/laRdeS6YIQKIys8pg+lRSRdVMTYjivs=",
		zipFileHash: "7b0a141b1b0b49772ab4eecfd11dfd6609a94a5e868cab04a3abb1861ffaa877",
	},
	{
		vcs:         "git",
		path:        "gopkg.in/check.v1",
		rev:         "20d25e280405",
		version:     "v1.0.0-20161208181325-20d25e280405",
		name:        "20d25e2804050c1cd24a7eea1e7a6447dd0e74ec",
		short:       "20d25e280405",
		time:        time.Date(2016, 12, 8, 18, 13, 25, 0, time.UTC),
		gomod:       "module gopkg.in/check.v1\n",
		zipSum:      "h1:829vOVxxusYHC+IqBtkX5mbKtsY9fheQiQn0MZRVLfQ=",
		zipFileHash: "9e7cb3f4f1e66d722306442b0dbe1f6f43d74d1736d54c510537bdfb1d6f432f",
	},
	{
		vcs:         "git",
		path:        "vcs-test.golang.org/go/mod/gitrepo1",
		rev:         "master",
		version:     "v1.2.4-annotated",
		name:        "ede458df7cd0fdca520df19a33158086a8a68e81",
		short:       "ede458df7cd0",
		time:        time.Date(2018, 4, 17, 19, 43, 22, 0, time.UTC),
		gomod:       "module vcs-test.golang.org/go/mod/gitrepo1\n",
		zipSum:      "h1:YJYZRsM9BHFTlVr8YADjT0cJH8uFIDtoc5NLiVqZEx8=",
		zipFileHash: "c15e49d58b7a4c37966cbe5bc01a0330cd5f2927e990e1839bda1d407766d9c5",
	},
	{
		vcs:         "git",
		path:        "gopkg.in/natefinch/lumberjack.v2",
		rev:         "latest",
		version:     "v2.0.0-20170531160350-a96e63847dc3",
		name:        "a96e63847dc3c67d17befa69c303767e2f84e54f",
		short:       "a96e63847dc3",
		time:        time.Date(2017, 5, 31, 16, 3, 50, 0, time.UTC),
		gomod:       "module gopkg.in/natefinch/lumberjack.v2\n",
		zipSum:      "h1:AFxeG48hTWHhDTQDk/m2gorfVHUEa9vo3tp3D7TzwjI=",
		zipFileHash: "b5de0da7bbbec76709eef1ac71b6c9ff423b9fbf3bb97b56743450d4937b06d5",
	},
	{
		vcs:  "git",
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
		vcs:         "git",
		path:        "vcs-test.golang.org/go/v2module/v2",
		rev:         "v2.0.0",
		version:     "v2.0.0",
		name:        "203b91c896acd173aa719e4cdcb7d463c4b090fa",
		short:       "203b91c896ac",
		time:        time.Date(2019, 4, 3, 15, 52, 15, 0, time.UTC),
		gomod:       "module vcs-test.golang.org/go/v2module/v2\n\ngo 1.12\n",
		zipSum:      "h1:JItBZ+gwA5WvtZEGEbuDL4lUttGtLrs53lmdurq3bOg=",
		zipFileHash: "9ea9ae1673cffcc44b7fdd3cc89953d68c102449b46c982dbf085e4f2e394da5",
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
					if tt.vcs != "mod" {
						testenv.MustHaveExecPath(t, tt.vcs)
					}

					repo, err := Lookup("direct", tt.path)
					if tt.lookErr != "" {
						if err != nil && err.Error() == tt.lookErr {
							return
						}
						t.Errorf("Lookup(%q): %v, want error %q", tt.path, err, tt.lookErr)
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

					if tt.gomod != "" || tt.gomodErr != "" {
						data, err := repo.GoMod(tt.version)
						if err != nil && tt.gomodErr == "" {
							t.Errorf("repo.GoMod(%q): %v", tt.version, err)
						} else if err != nil && tt.gomodErr != "" {
							if err.Error() != tt.gomodErr {
								t.Errorf("repo.GoMod(%q): %v, want %q", tt.version, err, tt.gomodErr)
							}
						} else if tt.gomodErr != "" {
							t.Errorf("repo.GoMod(%q) = %q, want error %q", tt.version, data, tt.gomodErr)
						} else if string(data) != tt.gomod {
							t.Errorf("repo.GoMod(%q) = %q, want %q", tt.version, data, tt.gomod)
						}
					}

					needHash := !testing.Short() && (tt.zipFileHash != "" || tt.zipSum != "")
					if tt.zip != nil || tt.zipErr != "" || needHash {
						f, err := ioutil.TempFile(tmpdir, tt.version+".zip.")
						if err != nil {
							t.Fatalf("ioutil.TempFile: %v", err)
						}
						zipfile := f.Name()
						defer func() {
							f.Close()
							os.Remove(zipfile)
						}()

						var w io.Writer
						var h hash.Hash
						if needHash {
							h = sha256.New()
							w = io.MultiWriter(f, h)
						} else {
							w = f
						}
						err = repo.Zip(w, tt.version)
						f.Close()
						if err != nil {
							if tt.zipErr != "" {
								if err.Error() == tt.zipErr {
									return
								}
								t.Fatalf("repo.Zip(%q): %v, want error %q", tt.version, err, tt.zipErr)
							}
							t.Fatalf("repo.Zip(%q): %v", tt.version, err)
						}
						if tt.zipErr != "" {
							t.Errorf("repo.Zip(%q): success, want error %q", tt.version, tt.zipErr)
						}

						if tt.zip != nil {
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

						if needHash {
							sum, err := dirhash.HashZip(zipfile, dirhash.Hash1)
							if err != nil {
								t.Errorf("repo.Zip(%q): %v", tt.version, err)
							} else if sum != tt.zipSum {
								t.Errorf("repo.Zip(%q): got file with sum %q, want %q", tt.version, sum, tt.zipSum)
							} else if zipFileHash := hex.EncodeToString(h.Sum(nil)); zipFileHash != tt.zipFileHash {
								t.Errorf("repo.Zip(%q): got file with hash %q, want %q (but content has correct sum)", tt.version, zipFileHash, tt.zipFileHash)
							}
						}
					}
				}
			}
			t.Run(strings.ReplaceAll(tt.path, "/", "_")+"/"+tt.rev, f(tt))
			if strings.HasPrefix(tt.path, vgotest1git) {
				for vcs, alt := range altVgotests {
					altTest := tt
					altTest.vcs = vcs
					altTest.path = alt + strings.TrimPrefix(altTest.path, vgotest1git)
					if strings.HasPrefix(altTest.mpath, vgotest1git) {
						altTest.mpath = alt + strings.TrimPrefix(altTest.mpath, vgotest1git)
					}
					var m map[string]string
					if alt == vgotest1hg {
						m = hgmap
					}
					altTest.version = remap(altTest.version, m)
					altTest.name = remap(altTest.name, m)
					altTest.short = remap(altTest.short, m)
					altTest.rev = remap(altTest.rev, m)
					altTest.err = remap(altTest.err, m)
					altTest.gomodErr = remap(altTest.gomodErr, m)
					altTest.zipErr = remap(altTest.zipErr, m)
					altTest.zipSum = ""
					altTest.zipFileHash = ""
					t.Run(strings.ReplaceAll(altTest.path, "/", "_")+"/"+altTest.rev, f(altTest))
				}
			}
		}
	})
}

var hgmap = map[string]string{
	"github.com/rsc/vgotest1":                  "vcs-test.golang.org/hg/vgotest1.hg",
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
	vcs      string
	path     string
	prefix   string
	versions []string
}{
	{
		vcs:      "git",
		path:     "github.com/rsc/vgotest1",
		versions: []string{"v0.0.0", "v0.0.1", "v1.0.0", "v1.0.1", "v1.0.2", "v1.0.3", "v1.1.0"},
	},
	{
		vcs:      "git",
		path:     "github.com/rsc/vgotest1",
		prefix:   "v1.0",
		versions: []string{"v1.0.0", "v1.0.1", "v1.0.2", "v1.0.3"},
	},
	{
		vcs:      "git",
		path:     "github.com/rsc/vgotest1/v2",
		versions: []string{"v2.0.0", "v2.0.1", "v2.0.2", "v2.0.3", "v2.0.4", "v2.0.5", "v2.0.6"},
	},
	{
		vcs:      "mod",
		path:     "swtch.com/testmod",
		versions: []string{"v1.0.0", "v1.1.1"},
	},
	{
		vcs:      "git",
		path:     "gopkg.in/russross/blackfriday.v2",
		versions: []string{"v2.0.0", "v2.0.1"},
	},
	{
		vcs:      "git",
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
				if tt.vcs != "mod" {
					testenv.MustHaveExecPath(t, tt.vcs)
				}

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
	vcs     string
	path    string
	version string
	err     string
}{
	{
		vcs:  "git",
		path: "github.com/rsc/empty",
		err:  "no commits",
	},
	{
		vcs:  "git",
		path: "github.com/rsc/vgotest1",
		err:  `github.com/rsc/vgotest1@v0.0.0-20180219223237-a08abb797a67: invalid version: go.mod has post-v0 module path "github.com/vgotest1/v2" at revision a08abb797a67`,
	},
	{
		vcs:  "git",
		path: "github.com/rsc/vgotest1/v2",
		err:  `github.com/rsc/vgotest1/v2@v2.0.0-20180219223237-a08abb797a67: invalid version: github.com/rsc/vgotest1/go.mod and .../v2/go.mod both have .../v2 module paths at revision a08abb797a67`,
	},
	{
		vcs:  "git",
		path: "github.com/rsc/vgotest1/subdir",
		err:  "github.com/rsc/vgotest1/subdir@v0.0.0-20180219223237-a08abb797a67: invalid version: missing github.com/rsc/vgotest1/subdir/go.mod at revision a08abb797a67",
	},
	{
		vcs:     "git",
		path:    "vcs-test.golang.org/git/commit-after-tag.git",
		version: "v1.0.1-0.20190715211727-b325d8217783",
	},
	{
		vcs:     "git",
		path:    "vcs-test.golang.org/git/no-tags.git",
		version: "v0.0.0-20190715212047-e706ba1d9f6d",
	},
	{
		vcs:     "mod",
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
				if tt.vcs != "mod" {
					testenv.MustHaveExecPath(t, tt.vcs)
				}

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
	codehost.Repo
}

func (ch *fixedTagsRepo) Tags(string) ([]string, error) { return ch.tags, nil }

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
