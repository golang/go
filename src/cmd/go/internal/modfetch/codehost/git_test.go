// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codehost

import (
	"archive/zip"
	"bytes"
	"flag"
	"internal/testenv"
	"io"
	"io/fs"
	"log"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestMain(m *testing.M) {
	// needed for initializing the test environment variables as testing.Short
	// and HasExternalNetwork
	flag.Parse()
	os.Exit(testMain(m))
}

const (
	gitrepo1 = "https://vcs-test.golang.org/git/gitrepo1"
	hgrepo1  = "https://vcs-test.golang.org/hg/hgrepo1"
)

var altRepos = []string{
	"localGitRepo",
	hgrepo1,
}

// TODO: Convert gitrepo1 to svn, bzr, fossil and add tests.
// For now, at least the hgrepo1 tests check the general vcs.go logic.

// localGitRepo is like gitrepo1 but allows archive access.
var localGitRepo string

func testMain(m *testing.M) int {
	dir, err := os.MkdirTemp("", "gitrepo-test-")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(dir)

	if testenv.HasExternalNetwork() && testenv.HasExec() {
		if _, err := exec.LookPath("git"); err == nil {
			// Clone gitrepo1 into a local directory.
			// If we use a file:// URL to access the local directory,
			// then git starts up all the usual protocol machinery,
			// which will let us test remote git archive invocations.
			localGitRepo = filepath.Join(dir, "gitrepo2")
			if _, err := Run("", "git", "clone", "--mirror", gitrepo1, localGitRepo); err != nil {
				log.Fatal(err)
			}
			if _, err := Run(localGitRepo, "git", "config", "daemon.uploadarch", "true"); err != nil {
				log.Fatal(err)
			}
		}
	}

	return m.Run()
}

func testRepo(t *testing.T, remote string) (Repo, error) {
	if remote == "localGitRepo" {
		// Convert absolute path to file URL. LocalGitRepo will not accept
		// Windows absolute paths because they look like a host:path remote.
		// TODO(golang.org/issue/32456): use url.FromFilePath when implemented.
		var url string
		if strings.HasPrefix(localGitRepo, "/") {
			url = "file://" + localGitRepo
		} else {
			url = "file:///" + filepath.ToSlash(localGitRepo)
		}
		testenv.MustHaveExecPath(t, "git")
		return LocalGitRepo(url)
	}
	vcs := "git"
	for _, k := range []string{"hg"} {
		if strings.Contains(remote, "/"+k+"/") {
			vcs = k
		}
	}
	testenv.MustHaveExecPath(t, vcs)
	return NewRepo(vcs, remote)
}

var tagsTests = []struct {
	repo   string
	prefix string
	tags   []string
}{
	{gitrepo1, "xxx", []string{}},
	{gitrepo1, "", []string{"v1.2.3", "v1.2.4-annotated", "v2.0.1", "v2.0.2", "v2.3"}},
	{gitrepo1, "v", []string{"v1.2.3", "v1.2.4-annotated", "v2.0.1", "v2.0.2", "v2.3"}},
	{gitrepo1, "v1", []string{"v1.2.3", "v1.2.4-annotated"}},
	{gitrepo1, "2", []string{}},
}

func TestTags(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)
	testenv.MustHaveExec(t)

	for _, tt := range tagsTests {
		f := func(t *testing.T) {
			r, err := testRepo(t, tt.repo)
			if err != nil {
				t.Fatal(err)
			}
			tags, err := r.Tags(tt.prefix)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(tags, tt.tags) {
				t.Errorf("Tags: incorrect tags\nhave %v\nwant %v", tags, tt.tags)
			}
		}
		t.Run(path.Base(tt.repo)+"/"+tt.prefix, f)
		if tt.repo == gitrepo1 {
			for _, tt.repo = range altRepos {
				t.Run(path.Base(tt.repo)+"/"+tt.prefix, f)
			}
		}
	}
}

var latestTests = []struct {
	repo string
	info *RevInfo
}{
	{
		gitrepo1,
		&RevInfo{
			Name:    "ede458df7cd0fdca520df19a33158086a8a68e81",
			Short:   "ede458df7cd0",
			Version: "ede458df7cd0fdca520df19a33158086a8a68e81",
			Time:    time.Date(2018, 4, 17, 19, 43, 22, 0, time.UTC),
			Tags:    []string{"v1.2.3", "v1.2.4-annotated"},
		},
	},
	{
		hgrepo1,
		&RevInfo{
			Name:    "18518c07eb8ed5c80221e997e518cccaa8c0c287",
			Short:   "18518c07eb8e",
			Version: "18518c07eb8ed5c80221e997e518cccaa8c0c287",
			Time:    time.Date(2018, 6, 27, 16, 16, 30, 0, time.UTC),
		},
	},
}

func TestLatest(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)
	testenv.MustHaveExec(t)

	for _, tt := range latestTests {
		f := func(t *testing.T) {
			r, err := testRepo(t, tt.repo)
			if err != nil {
				t.Fatal(err)
			}
			info, err := r.Latest()
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(info, tt.info) {
				t.Errorf("Latest: incorrect info\nhave %+v\nwant %+v", *info, *tt.info)
			}
		}
		t.Run(path.Base(tt.repo), f)
		if tt.repo == gitrepo1 {
			tt.repo = "localGitRepo"
			t.Run(path.Base(tt.repo), f)
		}
	}
}

var readFileTests = []struct {
	repo string
	rev  string
	file string
	err  string
	data string
}{
	{
		repo: gitrepo1,
		rev:  "latest",
		file: "README",
		data: "",
	},
	{
		repo: gitrepo1,
		rev:  "v2",
		file: "another.txt",
		data: "another\n",
	},
	{
		repo: gitrepo1,
		rev:  "v2.3.4",
		file: "another.txt",
		err:  fs.ErrNotExist.Error(),
	},
}

func TestReadFile(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)
	testenv.MustHaveExec(t)

	for _, tt := range readFileTests {
		f := func(t *testing.T) {
			r, err := testRepo(t, tt.repo)
			if err != nil {
				t.Fatal(err)
			}
			data, err := r.ReadFile(tt.rev, tt.file, 100)
			if err != nil {
				if tt.err == "" {
					t.Fatalf("ReadFile: unexpected error %v", err)
				}
				if !strings.Contains(err.Error(), tt.err) {
					t.Fatalf("ReadFile: wrong error %q, want %q", err, tt.err)
				}
				if len(data) != 0 {
					t.Errorf("ReadFile: non-empty data %q with error %v", data, err)
				}
				return
			}
			if tt.err != "" {
				t.Fatalf("ReadFile: no error, wanted %v", tt.err)
			}
			if string(data) != tt.data {
				t.Errorf("ReadFile: incorrect data\nhave %q\nwant %q", data, tt.data)
			}
		}
		t.Run(path.Base(tt.repo)+"/"+tt.rev+"/"+tt.file, f)
		if tt.repo == gitrepo1 {
			for _, tt.repo = range altRepos {
				t.Run(path.Base(tt.repo)+"/"+tt.rev+"/"+tt.file, f)
			}
		}
	}
}

var readZipTests = []struct {
	repo   string
	rev    string
	subdir string
	err    string
	files  map[string]uint64
}{
	{
		repo:   gitrepo1,
		rev:    "v2.3.4",
		subdir: "",
		files: map[string]uint64{
			"prefix/":       0,
			"prefix/README": 0,
			"prefix/v2":     3,
		},
	},
	{
		repo:   hgrepo1,
		rev:    "v2.3.4",
		subdir: "",
		files: map[string]uint64{
			"prefix/.hg_archival.txt": ^uint64(0),
			"prefix/README":           0,
			"prefix/v2":               3,
		},
	},

	{
		repo:   gitrepo1,
		rev:    "v2",
		subdir: "",
		files: map[string]uint64{
			"prefix/":            0,
			"prefix/README":      0,
			"prefix/v2":          3,
			"prefix/another.txt": 8,
			"prefix/foo.txt":     13,
		},
	},
	{
		repo:   hgrepo1,
		rev:    "v2",
		subdir: "",
		files: map[string]uint64{
			"prefix/.hg_archival.txt": ^uint64(0),
			"prefix/README":           0,
			"prefix/v2":               3,
			"prefix/another.txt":      8,
			"prefix/foo.txt":          13,
		},
	},

	{
		repo:   gitrepo1,
		rev:    "v3",
		subdir: "",
		files: map[string]uint64{
			"prefix/":                    0,
			"prefix/v3/":                 0,
			"prefix/v3/sub/":             0,
			"prefix/v3/sub/dir/":         0,
			"prefix/v3/sub/dir/file.txt": 16,
			"prefix/README":              0,
		},
	},
	{
		repo:   hgrepo1,
		rev:    "v3",
		subdir: "",
		files: map[string]uint64{
			"prefix/.hg_archival.txt":    ^uint64(0),
			"prefix/.hgtags":             405,
			"prefix/v3/sub/dir/file.txt": 16,
			"prefix/README":              0,
		},
	},

	{
		repo:   gitrepo1,
		rev:    "v3",
		subdir: "v3/sub/dir",
		files: map[string]uint64{
			"prefix/":                    0,
			"prefix/v3/":                 0,
			"prefix/v3/sub/":             0,
			"prefix/v3/sub/dir/":         0,
			"prefix/v3/sub/dir/file.txt": 16,
		},
	},
	{
		repo:   hgrepo1,
		rev:    "v3",
		subdir: "v3/sub/dir",
		files: map[string]uint64{
			"prefix/v3/sub/dir/file.txt": 16,
		},
	},

	{
		repo:   gitrepo1,
		rev:    "v3",
		subdir: "v3/sub",
		files: map[string]uint64{
			"prefix/":                    0,
			"prefix/v3/":                 0,
			"prefix/v3/sub/":             0,
			"prefix/v3/sub/dir/":         0,
			"prefix/v3/sub/dir/file.txt": 16,
		},
	},
	{
		repo:   hgrepo1,
		rev:    "v3",
		subdir: "v3/sub",
		files: map[string]uint64{
			"prefix/v3/sub/dir/file.txt": 16,
		},
	},

	{
		repo:   gitrepo1,
		rev:    "aaaaaaaaab",
		subdir: "",
		err:    "unknown revision",
	},
	{
		repo:   hgrepo1,
		rev:    "aaaaaaaaab",
		subdir: "",
		err:    "unknown revision",
	},

	{
		repo:   "https://github.com/rsc/vgotest1",
		rev:    "submod/v1.0.4",
		subdir: "submod",
		files: map[string]uint64{
			"prefix/":                0,
			"prefix/submod/":         0,
			"prefix/submod/go.mod":   53,
			"prefix/submod/pkg/":     0,
			"prefix/submod/pkg/p.go": 31,
		},
	},
}

type zipFile struct {
	name string
	size int64
}

func TestReadZip(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)
	testenv.MustHaveExec(t)

	for _, tt := range readZipTests {
		f := func(t *testing.T) {
			r, err := testRepo(t, tt.repo)
			if err != nil {
				t.Fatal(err)
			}
			rc, err := r.ReadZip(tt.rev, tt.subdir, 100000)
			if err != nil {
				if tt.err == "" {
					t.Fatalf("ReadZip: unexpected error %v", err)
				}
				if !strings.Contains(err.Error(), tt.err) {
					t.Fatalf("ReadZip: wrong error %q, want %q", err, tt.err)
				}
				if rc != nil {
					t.Errorf("ReadZip: non-nil io.ReadCloser with error %v", err)
				}
				return
			}
			defer rc.Close()
			if tt.err != "" {
				t.Fatalf("ReadZip: no error, wanted %v", tt.err)
			}
			zipdata, err := io.ReadAll(rc)
			if err != nil {
				t.Fatal(err)
			}
			z, err := zip.NewReader(bytes.NewReader(zipdata), int64(len(zipdata)))
			if err != nil {
				t.Fatalf("ReadZip: cannot read zip file: %v", err)
			}
			have := make(map[string]bool)
			for _, f := range z.File {
				size, ok := tt.files[f.Name]
				if !ok {
					t.Errorf("ReadZip: unexpected file %s", f.Name)
					continue
				}
				have[f.Name] = true
				if size != ^uint64(0) && f.UncompressedSize64 != size {
					t.Errorf("ReadZip: file %s has unexpected size %d != %d", f.Name, f.UncompressedSize64, size)
				}
			}
			for name := range tt.files {
				if !have[name] {
					t.Errorf("ReadZip: missing file %s", name)
				}
			}
		}
		t.Run(path.Base(tt.repo)+"/"+tt.rev+"/"+tt.subdir, f)
		if tt.repo == gitrepo1 {
			tt.repo = "localGitRepo"
			t.Run(path.Base(tt.repo)+"/"+tt.rev+"/"+tt.subdir, f)
		}
	}
}

var hgmap = map[string]string{
	"HEAD": "41964ddce1180313bdc01d0a39a2813344d6261d", // not tip due to bad hgrepo1 conversion
	"9d02800338b8a55be062c838d1f02e0c5780b9eb": "8f49ee7a6ddcdec6f0112d9dca48d4a2e4c3c09e",
	"76a00fb249b7f93091bc2c89a789dab1fc1bc26f": "88fde824ec8b41a76baa16b7e84212cee9f3edd0",
	"ede458df7cd0fdca520df19a33158086a8a68e81": "41964ddce1180313bdc01d0a39a2813344d6261d",
	"97f6aa59c81c623494825b43d39e445566e429a4": "c0cbbfb24c7c3c50c35c7b88e7db777da4ff625d",
}

var statTests = []struct {
	repo string
	rev  string
	err  string
	info *RevInfo
}{
	{
		repo: gitrepo1,
		rev:  "HEAD",
		info: &RevInfo{
			Name:    "ede458df7cd0fdca520df19a33158086a8a68e81",
			Short:   "ede458df7cd0",
			Version: "ede458df7cd0fdca520df19a33158086a8a68e81",
			Time:    time.Date(2018, 4, 17, 19, 43, 22, 0, time.UTC),
			Tags:    []string{"v1.2.3", "v1.2.4-annotated"},
		},
	},
	{
		repo: gitrepo1,
		rev:  "v2", // branch
		info: &RevInfo{
			Name:    "9d02800338b8a55be062c838d1f02e0c5780b9eb",
			Short:   "9d02800338b8",
			Version: "9d02800338b8a55be062c838d1f02e0c5780b9eb",
			Time:    time.Date(2018, 4, 17, 20, 00, 32, 0, time.UTC),
			Tags:    []string{"v2.0.2"},
		},
	},
	{
		repo: gitrepo1,
		rev:  "v2.3.4", // badly-named branch (semver should be a tag)
		info: &RevInfo{
			Name:    "76a00fb249b7f93091bc2c89a789dab1fc1bc26f",
			Short:   "76a00fb249b7",
			Version: "76a00fb249b7f93091bc2c89a789dab1fc1bc26f",
			Time:    time.Date(2018, 4, 17, 19, 45, 48, 0, time.UTC),
			Tags:    []string{"v2.0.1", "v2.3"},
		},
	},
	{
		repo: gitrepo1,
		rev:  "v2.3", // badly-named tag (we only respect full semver v2.3.0)
		info: &RevInfo{
			Name:    "76a00fb249b7f93091bc2c89a789dab1fc1bc26f",
			Short:   "76a00fb249b7",
			Version: "v2.3",
			Time:    time.Date(2018, 4, 17, 19, 45, 48, 0, time.UTC),
			Tags:    []string{"v2.0.1", "v2.3"},
		},
	},
	{
		repo: gitrepo1,
		rev:  "v1.2.3", // tag
		info: &RevInfo{
			Name:    "ede458df7cd0fdca520df19a33158086a8a68e81",
			Short:   "ede458df7cd0",
			Version: "v1.2.3",
			Time:    time.Date(2018, 4, 17, 19, 43, 22, 0, time.UTC),
			Tags:    []string{"v1.2.3", "v1.2.4-annotated"},
		},
	},
	{
		repo: gitrepo1,
		rev:  "ede458df", // hash prefix in refs
		info: &RevInfo{
			Name:    "ede458df7cd0fdca520df19a33158086a8a68e81",
			Short:   "ede458df7cd0",
			Version: "ede458df7cd0fdca520df19a33158086a8a68e81",
			Time:    time.Date(2018, 4, 17, 19, 43, 22, 0, time.UTC),
			Tags:    []string{"v1.2.3", "v1.2.4-annotated"},
		},
	},
	{
		repo: gitrepo1,
		rev:  "97f6aa59", // hash prefix not in refs
		info: &RevInfo{
			Name:    "97f6aa59c81c623494825b43d39e445566e429a4",
			Short:   "97f6aa59c81c",
			Version: "97f6aa59c81c623494825b43d39e445566e429a4",
			Time:    time.Date(2018, 4, 17, 20, 0, 19, 0, time.UTC),
		},
	},
	{
		repo: gitrepo1,
		rev:  "v1.2.4-annotated", // annotated tag uses unwrapped commit hash
		info: &RevInfo{
			Name:    "ede458df7cd0fdca520df19a33158086a8a68e81",
			Short:   "ede458df7cd0",
			Version: "v1.2.4-annotated",
			Time:    time.Date(2018, 4, 17, 19, 43, 22, 0, time.UTC),
			Tags:    []string{"v1.2.3", "v1.2.4-annotated"},
		},
	},
	{
		repo: gitrepo1,
		rev:  "aaaaaaaaab",
		err:  "unknown revision",
	},
}

func TestStat(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)
	testenv.MustHaveExec(t)

	for _, tt := range statTests {
		f := func(t *testing.T) {
			r, err := testRepo(t, tt.repo)
			if err != nil {
				t.Fatal(err)
			}
			info, err := r.Stat(tt.rev)
			if err != nil {
				if tt.err == "" {
					t.Fatalf("Stat: unexpected error %v", err)
				}
				if !strings.Contains(err.Error(), tt.err) {
					t.Fatalf("Stat: wrong error %q, want %q", err, tt.err)
				}
				if info != nil {
					t.Errorf("Stat: non-nil info with error %q", err)
				}
				return
			}
			if !reflect.DeepEqual(info, tt.info) {
				t.Errorf("Stat: incorrect info\nhave %+v\nwant %+v", *info, *tt.info)
			}
		}
		t.Run(path.Base(tt.repo)+"/"+tt.rev, f)
		if tt.repo == gitrepo1 {
			for _, tt.repo = range altRepos {
				old := tt
				var m map[string]string
				if tt.repo == hgrepo1 {
					m = hgmap
				}
				if tt.info != nil {
					info := *tt.info
					tt.info = &info
					tt.info.Name = remap(tt.info.Name, m)
					tt.info.Version = remap(tt.info.Version, m)
					tt.info.Short = remap(tt.info.Short, m)
				}
				tt.rev = remap(tt.rev, m)
				t.Run(path.Base(tt.repo)+"/"+tt.rev, f)
				tt = old
			}
		}
	}
}

func remap(name string, m map[string]string) string {
	if m[name] != "" {
		return m[name]
	}
	if AllHex(name) {
		for k, v := range m {
			if strings.HasPrefix(k, name) {
				return v[:len(name)]
			}
		}
	}
	return name
}
