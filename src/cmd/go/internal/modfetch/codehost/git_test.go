// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codehost

import (
	"archive/zip"
	"bytes"
	"cmd/go/internal/cfg"
	"cmd/go/internal/vcweb/vcstest"
	"context"
	"flag"
	"internal/testenv"
	"io"
	"io/fs"
	"log"
	"os"
	"path"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestMain(m *testing.M) {
	flag.Parse()
	if err := testMain(m); err != nil {
		log.Fatal(err)
	}
}

var gitrepo1, hgrepo1, vgotest1 string

var altRepos = func() []string {
	return []string{
		"localGitRepo",
		hgrepo1,
	}
}

// TODO: Convert gitrepo1 to svn, bzr, fossil and add tests.
// For now, at least the hgrepo1 tests check the general vcs.go logic.

// localGitRepo is like gitrepo1 but allows archive access
// (although that doesn't really matter after CL 120041),
// and has a file:// URL instead of http:// or https://
// (which might still matter).
var localGitRepo string

// localGitURL initializes the repo in localGitRepo and returns its URL.
func localGitURL(t testing.TB) string {
	testenv.MustHaveExecPath(t, "git")
	if runtime.GOOS == "android" && strings.HasSuffix(testenv.Builder(), "-corellium") {
		testenv.SkipFlaky(t, 59940)
	}

	localGitURLOnce.Do(func() {
		// Clone gitrepo1 into a local directory.
		// If we use a file:// URL to access the local directory,
		// then git starts up all the usual protocol machinery,
		// which will let us test remote git archive invocations.
		_, localGitURLErr = Run(context.Background(), "", "git", "clone", "--mirror", gitrepo1, localGitRepo)
		if localGitURLErr != nil {
			return
		}
		repo := gitRepo{dir: localGitRepo}
		_, localGitURLErr = repo.runGit(context.Background(), "git", "config", "daemon.uploadarch", "true")
	})

	if localGitURLErr != nil {
		t.Fatal(localGitURLErr)
	}
	// Convert absolute path to file URL. LocalGitRepo will not accept
	// Windows absolute paths because they look like a host:path remote.
	// TODO(golang.org/issue/32456): use url.FromFilePath when implemented.
	if strings.HasPrefix(localGitRepo, "/") {
		return "file://" + localGitRepo
	} else {
		return "file:///" + filepath.ToSlash(localGitRepo)
	}
}

var (
	localGitURLOnce sync.Once
	localGitURLErr  error
)

func testMain(m *testing.M) (err error) {
	cfg.BuildX = testing.Verbose()

	srv, err := vcstest.NewServer()
	if err != nil {
		return err
	}
	defer func() {
		if closeErr := srv.Close(); err == nil {
			err = closeErr
		}
	}()

	gitrepo1 = srv.HTTP.URL + "/git/gitrepo1"
	hgrepo1 = srv.HTTP.URL + "/hg/hgrepo1"
	vgotest1 = srv.HTTP.URL + "/git/vgotest1"

	dir, err := os.MkdirTemp("", "gitrepo-test-")
	if err != nil {
		return err
	}
	defer func() {
		if rmErr := os.RemoveAll(dir); err == nil {
			err = rmErr
		}
	}()

	localGitRepo = filepath.Join(dir, "gitrepo2")

	// Redirect the module cache to a fresh directory to avoid crosstalk, and make
	// it read/write so that the test can still clean it up easily when done.
	cfg.GOMODCACHE = filepath.Join(dir, "modcache")
	cfg.ModCacheRW = true

	m.Run()
	return nil
}

func testContext(t testing.TB) context.Context {
	w := newTestWriter(t)
	return cfg.WithBuildXWriter(context.Background(), w)
}

// A testWriter is an io.Writer that writes to a test's log.
//
// The writer batches written data until the last byte of a write is a newline
// character, then flushes the batched data as a single call to Logf.
// Any remaining unflushed data is logged during Cleanup.
type testWriter struct {
	t testing.TB

	mu  sync.Mutex
	buf bytes.Buffer
}

func newTestWriter(t testing.TB) *testWriter {
	w := &testWriter{t: t}

	t.Cleanup(func() {
		w.mu.Lock()
		defer w.mu.Unlock()
		if b := w.buf.Bytes(); len(b) > 0 {
			w.t.Logf("%s", b)
			w.buf.Reset()
		}
	})

	return w
}

func (w *testWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	n, err := w.buf.Write(p)
	if b := w.buf.Bytes(); len(b) > 0 && b[len(b)-1] == '\n' {
		w.t.Logf("%s", b)
		w.buf.Reset()
	}
	return n, err
}

func testRepo(ctx context.Context, t *testing.T, remote string) (Repo, error) {
	if remote == "localGitRepo" {
		return NewRepo(ctx, "git", localGitURL(t), false)
	}
	vcsName := "git"
	for _, k := range []string{"hg"} {
		if strings.Contains(remote, "/"+k+"/") {
			vcsName = k
		}
	}
	if testing.Short() && vcsName == "hg" {
		t.Skipf("skipping hg test in short mode: hg is slow")
	}
	testenv.MustHaveExecPath(t, vcsName)
	if runtime.GOOS == "android" && strings.HasSuffix(testenv.Builder(), "-corellium") {
		testenv.SkipFlaky(t, 59940)
	}
	return NewRepo(ctx, vcsName, remote, false)
}

func TestTags(t *testing.T) {
	t.Parallel()

	type tagsTest struct {
		repo   string
		prefix string
		tags   []Tag
	}

	runTest := func(tt tagsTest) func(*testing.T) {
		return func(t *testing.T) {
			t.Parallel()
			ctx := testContext(t)

			r, err := testRepo(ctx, t, tt.repo)
			if err != nil {
				t.Fatal(err)
			}
			tags, err := r.Tags(ctx, tt.prefix)
			if err != nil {
				t.Fatal(err)
			}
			if tags == nil || !reflect.DeepEqual(tags.List, tt.tags) {
				t.Errorf("Tags(%q): incorrect tags\nhave %v\nwant %v", tt.prefix, tags, tt.tags)
			}
		}
	}

	for _, tt := range []tagsTest{
		{gitrepo1, "xxx", []Tag{}},
		{gitrepo1, "", []Tag{
			{"v1.2.3", "ede458df7cd0fdca520df19a33158086a8a68e81"},
			{"v1.2.4-annotated", "ede458df7cd0fdca520df19a33158086a8a68e81"},
			{"v2.0.1", "76a00fb249b7f93091bc2c89a789dab1fc1bc26f"},
			{"v2.0.2", "9d02800338b8a55be062c838d1f02e0c5780b9eb"},
			{"v2.3", "76a00fb249b7f93091bc2c89a789dab1fc1bc26f"},
		}},
		{gitrepo1, "v", []Tag{
			{"v1.2.3", "ede458df7cd0fdca520df19a33158086a8a68e81"},
			{"v1.2.4-annotated", "ede458df7cd0fdca520df19a33158086a8a68e81"},
			{"v2.0.1", "76a00fb249b7f93091bc2c89a789dab1fc1bc26f"},
			{"v2.0.2", "9d02800338b8a55be062c838d1f02e0c5780b9eb"},
			{"v2.3", "76a00fb249b7f93091bc2c89a789dab1fc1bc26f"},
		}},
		{gitrepo1, "v1", []Tag{
			{"v1.2.3", "ede458df7cd0fdca520df19a33158086a8a68e81"},
			{"v1.2.4-annotated", "ede458df7cd0fdca520df19a33158086a8a68e81"},
		}},
		{gitrepo1, "2", []Tag{}},
	} {
		t.Run(path.Base(tt.repo)+"/"+tt.prefix, runTest(tt))
		if tt.repo == gitrepo1 {
			// Clear hashes.
			clearTags := []Tag{}
			for _, tag := range tt.tags {
				clearTags = append(clearTags, Tag{tag.Name, ""})
			}
			tags := tt.tags
			for _, tt.repo = range altRepos() {
				if strings.Contains(tt.repo, "Git") {
					tt.tags = tags
				} else {
					tt.tags = clearTags
				}
				t.Run(path.Base(tt.repo)+"/"+tt.prefix, runTest(tt))
			}
		}
	}
}

func TestLatest(t *testing.T) {
	t.Parallel()

	type latestTest struct {
		repo string
		info *RevInfo
	}
	runTest := func(tt latestTest) func(*testing.T) {
		return func(t *testing.T) {
			t.Parallel()
			ctx := testContext(t)

			r, err := testRepo(ctx, t, tt.repo)
			if err != nil {
				t.Fatal(err)
			}
			info, err := r.Latest(ctx)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(info, tt.info) {
				t.Errorf("Latest: incorrect info\nhave %+v (origin %+v)\nwant %+v (origin %+v)", info, info.Origin, tt.info, tt.info.Origin)
			}
		}
	}

	for _, tt := range []latestTest{
		{
			gitrepo1,
			&RevInfo{
				Origin: &Origin{
					VCS:  "git",
					URL:  gitrepo1,
					Ref:  "HEAD",
					Hash: "ede458df7cd0fdca520df19a33158086a8a68e81",
				},
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
				Origin: &Origin{
					VCS:  "hg",
					URL:  hgrepo1,
					Hash: "18518c07eb8ed5c80221e997e518cccaa8c0c287",
				},
				Name:    "18518c07eb8ed5c80221e997e518cccaa8c0c287",
				Short:   "18518c07eb8e",
				Version: "18518c07eb8ed5c80221e997e518cccaa8c0c287",
				Time:    time.Date(2018, 6, 27, 16, 16, 30, 0, time.UTC),
			},
		},
	} {
		t.Run(path.Base(tt.repo), runTest(tt))
		if tt.repo == gitrepo1 {
			tt.repo = "localGitRepo"
			info := *tt.info
			tt.info = &info
			o := *info.Origin
			info.Origin = &o
			o.URL = localGitURL(t)
			t.Run(path.Base(tt.repo), runTest(tt))
		}
	}
}

func TestReadFile(t *testing.T) {
	t.Parallel()

	type readFileTest struct {
		repo string
		rev  string
		file string
		err  string
		data string
	}
	runTest := func(tt readFileTest) func(*testing.T) {
		return func(t *testing.T) {
			t.Parallel()
			ctx := testContext(t)

			r, err := testRepo(ctx, t, tt.repo)
			if err != nil {
				t.Fatal(err)
			}
			data, err := r.ReadFile(ctx, tt.rev, tt.file, 100)
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
	}

	for _, tt := range []readFileTest{
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
	} {
		t.Run(path.Base(tt.repo)+"/"+tt.rev+"/"+tt.file, runTest(tt))
		if tt.repo == gitrepo1 {
			for _, tt.repo = range altRepos() {
				t.Run(path.Base(tt.repo)+"/"+tt.rev+"/"+tt.file, runTest(tt))
			}
		}
	}
}

type zipFile struct {
	name string
	size int64
}

func TestReadZip(t *testing.T) {
	t.Parallel()

	type readZipTest struct {
		repo   string
		rev    string
		subdir string
		err    string
		files  map[string]uint64
	}
	runTest := func(tt readZipTest) func(*testing.T) {
		return func(t *testing.T) {
			t.Parallel()
			ctx := testContext(t)

			r, err := testRepo(ctx, t, tt.repo)
			if err != nil {
				t.Fatal(err)
			}
			rc, err := r.ReadZip(ctx, tt.rev, tt.subdir, 100000)
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
	}

	for _, tt := range []readZipTest{
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
			repo:   vgotest1,
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
	} {
		t.Run(path.Base(tt.repo)+"/"+tt.rev+"/"+tt.subdir, runTest(tt))
		if tt.repo == gitrepo1 {
			tt.repo = "localGitRepo"
			t.Run(path.Base(tt.repo)+"/"+tt.rev+"/"+tt.subdir, runTest(tt))
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

func TestStat(t *testing.T) {
	t.Parallel()

	type statTest struct {
		repo string
		rev  string
		err  string
		info *RevInfo
	}
	runTest := func(tt statTest) func(*testing.T) {
		return func(t *testing.T) {
			t.Parallel()
			ctx := testContext(t)

			r, err := testRepo(ctx, t, tt.repo)
			if err != nil {
				t.Fatal(err)
			}
			info, err := r.Stat(ctx, tt.rev)
			if err != nil {
				if tt.err == "" {
					t.Fatalf("Stat: unexpected error %v", err)
				}
				if !strings.Contains(err.Error(), tt.err) {
					t.Fatalf("Stat: wrong error %q, want %q", err, tt.err)
				}
				if info != nil && info.Origin == nil {
					t.Errorf("Stat: non-nil info with nil Origin with error %q", err)
				}
				return
			}
			info.Origin = nil // TestLatest and ../../../testdata/script/reuse_git.txt test Origin well enough
			if !reflect.DeepEqual(info, tt.info) {
				t.Errorf("Stat: incorrect info\nhave %+v\nwant %+v", *info, *tt.info)
			}
		}
	}

	for _, tt := range []statTest{
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
	} {
		t.Run(path.Base(tt.repo)+"/"+tt.rev, runTest(tt))
		if tt.repo == gitrepo1 {
			for _, tt.repo = range altRepos() {
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
				t.Run(path.Base(tt.repo)+"/"+tt.rev, runTest(tt))
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
