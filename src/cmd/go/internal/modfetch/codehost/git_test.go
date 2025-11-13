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
	"os/exec"
	"path"
	"path/filepath"
	"reflect"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"golang.org/x/mod/semver"
)

func TestMain(m *testing.M) {
	flag.Parse()
	if err := testMain(m); err != nil {
		log.Fatal(err)
	}
}

var gitrepo1, gitsha256repo, hgrepo1, vgotest1 string

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
		// TODO(david.finkel): do the same with the git repo using sha256 object hashes
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
	gitsha256repo = srv.HTTP.URL + "/git/gitrepo-sha256"
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

var gitVersLineExtract = regexp.MustCompile(`git version\s+([\d.]+)`)

func gitVersion(t testing.TB) string {
	gitOut, runErr := exec.Command("git", "version").CombinedOutput()
	if runErr != nil {
		t.Logf("failed to execute git version: %s", runErr)
		return "v0"
	}
	matches := gitVersLineExtract.FindSubmatch(gitOut)
	if len(matches) < 2 {
		t.Logf("git version extraction regexp did not match version line: %q", gitOut)
		return "v0"
	}
	return "v" + string(matches[1])
}

const minGitSHA256Vers = "v2.29"

func TestTags(t *testing.T) {
	t.Parallel()

	gitVers := gitVersion(t)

	type tagsTest struct {
		repo   string
		prefix string
		tags   []Tag
	}

	runTest := func(tt tagsTest) func(*testing.T) {
		return func(t *testing.T) {
			t.Parallel()
			if tt.repo == gitsha256repo && semver.Compare(gitVers, minGitSHA256Vers) < 0 {
				t.Skipf("git version is too old (%+v); skipping git sha256 test", gitVers)
			}
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
		{gitsha256repo, "xxx", []Tag{}},
		{gitsha256repo, "", []Tag{
			{"v1.2.3", "47b8b51b2a2d9d5caa3d460096c4e01f05700ce3a9390deb54400bd508214c5c"},
			{"v1.2.4-annotated", "47b8b51b2a2d9d5caa3d460096c4e01f05700ce3a9390deb54400bd508214c5c"},
			{"v1.3.0", "a9157cad2aa6dc2f78aa31fced5887f04e758afa8703f04d0178702ebf04ee17"},
			{"v2.0.1", "b7550fd9d2129c724c39ae0536e8b2fae4364d8c82bb8b0880c9b71f67295d09"},
			{"v2.0.2", "1401e4e1fdb4169b51d44a1ff62af63ccc708bf5c12d15051268b51bbb6cbd82"},
			{"v2.3", "b7550fd9d2129c724c39ae0536e8b2fae4364d8c82bb8b0880c9b71f67295d09"},
		}},
		{gitsha256repo, "v", []Tag{
			{"v1.2.3", "47b8b51b2a2d9d5caa3d460096c4e01f05700ce3a9390deb54400bd508214c5c"},
			{"v1.2.4-annotated", "47b8b51b2a2d9d5caa3d460096c4e01f05700ce3a9390deb54400bd508214c5c"},
			{"v1.3.0", "a9157cad2aa6dc2f78aa31fced5887f04e758afa8703f04d0178702ebf04ee17"},
			{"v2.0.1", "b7550fd9d2129c724c39ae0536e8b2fae4364d8c82bb8b0880c9b71f67295d09"},
			{"v2.0.2", "1401e4e1fdb4169b51d44a1ff62af63ccc708bf5c12d15051268b51bbb6cbd82"},
			{"v2.3", "b7550fd9d2129c724c39ae0536e8b2fae4364d8c82bb8b0880c9b71f67295d09"},
		}},
		{gitsha256repo, "v1", []Tag{
			{"v1.2.3", "47b8b51b2a2d9d5caa3d460096c4e01f05700ce3a9390deb54400bd508214c5c"},
			{"v1.2.4-annotated", "47b8b51b2a2d9d5caa3d460096c4e01f05700ce3a9390deb54400bd508214c5c"},
			{"v1.3.0", "a9157cad2aa6dc2f78aa31fced5887f04e758afa8703f04d0178702ebf04ee17"},
		}},
		{gitsha256repo, "2", []Tag{}},
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

	gitVers := gitVersion(t)

	type latestTest struct {
		repo string
		info *RevInfo
	}
	runTest := func(tt latestTest) func(*testing.T) {
		return func(t *testing.T) {
			t.Parallel()
			ctx := testContext(t)

			if tt.repo == gitsha256repo && semver.Compare(gitVers, minGitSHA256Vers) < 0 {
				t.Skipf("git version is too old (%+v); skipping git sha256 test", gitVers)
			}

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
			gitsha256repo,
			&RevInfo{
				Origin: &Origin{
					VCS:  "git",
					URL:  gitsha256repo,
					Ref:  "HEAD",
					Hash: "47b8b51b2a2d9d5caa3d460096c4e01f05700ce3a9390deb54400bd508214c5c",
				},
				Name:    "47b8b51b2a2d9d5caa3d460096c4e01f05700ce3a9390deb54400bd508214c5c",
				Short:   "47b8b51b2a2d",
				Version: "47b8b51b2a2d9d5caa3d460096c4e01f05700ce3a9390deb54400bd508214c5c",
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
					Ref:  "tip",
					Hash: "745aacc8b24decc44ac2b13870f5472b479f4d72",
				},
				Name:    "745aacc8b24decc44ac2b13870f5472b479f4d72",
				Short:   "745aacc8b24d",
				Version: "745aacc8b24decc44ac2b13870f5472b479f4d72",
				Time:    time.Date(2018, 6, 27, 16, 16, 10, 0, time.UTC),
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

	gitVers := gitVersion(t)

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

			if tt.repo == gitsha256repo && semver.Compare(gitVers, minGitSHA256Vers) < 0 {
				t.Skipf("git version is too old (%+v); skipping git sha256 test", gitVers)
			}

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
		{
			repo: gitsha256repo,
			rev:  "latest",
			file: "README",
			data: "",
		},
		{
			repo: gitsha256repo,
			rev:  "v2",
			file: "another.txt",
			data: "another\n",
		},
		{
			repo: gitsha256repo,
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

	gitVers := gitVersion(t)

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

			if tt.repo == gitsha256repo && semver.Compare(gitVers, minGitSHA256Vers) < 0 {
				t.Skipf("git version is too old (%+v); skipping git sha256 test", gitVers)
			}

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
			repo:   gitsha256repo,
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
			repo:   gitsha256repo,
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
			repo:   gitsha256repo,
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
			repo:   gitsha256repo,
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
			repo:   gitsha256repo,
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
			repo:   gitsha256repo,
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
	"HEAD": "c0186fb00e50985709b12266419f50bf11860166",
	"9d02800338b8a55be062c838d1f02e0c5780b9eb": "b1ed98abc2683d326f89b924875bf14bd584898e", // v2.0.2, v2
	"76a00fb249b7f93091bc2c89a789dab1fc1bc26f": "a546811101e11d6aff2ac72072d2d439b3a88f33", // v2.3, v2.0.1
	"ede458df7cd0fdca520df19a33158086a8a68e81": "c0186fb00e50985709b12266419f50bf11860166", // v1.2.3, v1.2.4-annotated
	"97f6aa59c81c623494825b43d39e445566e429a4": "c1638e3673b121d9c83e92166fce2a25dcadd6cb", // foo.txt commit on v2.3.4 branch
}

func TestStat(t *testing.T) {
	t.Parallel()

	gitVers := gitVersion(t)

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

			if tt.repo == gitsha256repo && semver.Compare(gitVers, minGitSHA256Vers) < 0 {
				t.Skipf("git version is too old (%+v); skipping git sha256 test", gitVers)
			}

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
		{
			repo: gitsha256repo,
			rev:  "HEAD",
			info: &RevInfo{
				Name:    "47b8b51b2a2d9d5caa3d460096c4e01f05700ce3a9390deb54400bd508214c5c",
				Short:   "47b8b51b2a2d",
				Version: "47b8b51b2a2d9d5caa3d460096c4e01f05700ce3a9390deb54400bd508214c5c",
				Time:    time.Date(2018, 4, 17, 19, 43, 22, 0, time.UTC),
				Tags:    []string{"v1.2.3", "v1.2.4-annotated"},
			},
		},
		{
			repo: gitsha256repo,
			rev:  "v2", // branch
			info: &RevInfo{
				Name:    "1401e4e1fdb4169b51d44a1ff62af63ccc708bf5c12d15051268b51bbb6cbd82",
				Short:   "1401e4e1fdb4",
				Version: "1401e4e1fdb4169b51d44a1ff62af63ccc708bf5c12d15051268b51bbb6cbd82",
				Time:    time.Date(2018, 4, 17, 20, 00, 32, 0, time.UTC),
				Tags:    []string{"v2.0.2"},
			},
		},
		{
			repo: gitsha256repo,
			rev:  "v2.3.4", // badly-named branch (semver should be a tag)
			info: &RevInfo{
				Name:    "b7550fd9d2129c724c39ae0536e8b2fae4364d8c82bb8b0880c9b71f67295d09",
				Short:   "b7550fd9d212",
				Version: "b7550fd9d2129c724c39ae0536e8b2fae4364d8c82bb8b0880c9b71f67295d09",
				Time:    time.Date(2018, 4, 17, 19, 45, 48, 0, time.UTC),
				Tags:    []string{"v2.0.1", "v2.3"},
			},
		},
		{
			repo: gitsha256repo,
			rev:  "v2.3", // badly-named tag (we only respect full semver v2.3.0)
			info: &RevInfo{
				Name:    "b7550fd9d2129c724c39ae0536e8b2fae4364d8c82bb8b0880c9b71f67295d09",
				Short:   "b7550fd9d212",
				Version: "v2.3",
				Time:    time.Date(2018, 4, 17, 19, 45, 48, 0, time.UTC),
				Tags:    []string{"v2.0.1", "v2.3"},
			},
		},
		{
			repo: gitsha256repo,
			rev:  "v1.2.3", // tag
			info: &RevInfo{
				Name:    "47b8b51b2a2d9d5caa3d460096c4e01f05700ce3a9390deb54400bd508214c5c",
				Short:   "47b8b51b2a2d",
				Version: "v1.2.3",
				Time:    time.Date(2018, 4, 17, 19, 43, 22, 0, time.UTC),
				Tags:    []string{"v1.2.3", "v1.2.4-annotated"},
			},
		},
		{
			repo: gitsha256repo,
			rev:  "47b8b51b", // hash prefix in refs
			info: &RevInfo{
				Name:    "47b8b51b2a2d9d5caa3d460096c4e01f05700ce3a9390deb54400bd508214c5c",
				Short:   "47b8b51b2a2d",
				Version: "47b8b51b2a2d9d5caa3d460096c4e01f05700ce3a9390deb54400bd508214c5c",
				Time:    time.Date(2018, 4, 17, 19, 43, 22, 0, time.UTC),
				Tags:    []string{"v1.2.3", "v1.2.4-annotated"},
			},
		},
		{
			repo: gitsha256repo,
			rev:  "0be440b6", // hash prefix not in refs
			info: &RevInfo{
				Name:    "0be440b60b6c81be26c7256781d8e57112ec46c8cd1a9481a8e78a283f10570c",
				Short:   "0be440b60b6c",
				Version: "0be440b60b6c81be26c7256781d8e57112ec46c8cd1a9481a8e78a283f10570c",
				Time:    time.Date(2018, 4, 17, 20, 0, 19, 0, time.UTC),
			},
		},
		{
			repo: gitsha256repo,
			rev:  "v1.2.4-annotated", // annotated tag uses unwrapped commit hash
			info: &RevInfo{
				Name:    "47b8b51b2a2d9d5caa3d460096c4e01f05700ce3a9390deb54400bd508214c5c",
				Short:   "47b8b51b2a2d",
				Version: "v1.2.4-annotated",
				Time:    time.Date(2018, 4, 17, 19, 43, 22, 0, time.UTC),
				Tags:    []string{"v1.2.3", "v1.2.4-annotated"},
			},
		},
		{
			repo: gitsha256repo,
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
