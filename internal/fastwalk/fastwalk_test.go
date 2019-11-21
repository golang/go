// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fastwalk_test

import (
	"bytes"
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"sync"
	"testing"

	"golang.org/x/tools/internal/fastwalk"
)

func formatFileModes(m map[string]os.FileMode) string {
	var keys []string
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var buf bytes.Buffer
	for _, k := range keys {
		fmt.Fprintf(&buf, "%-20s: %v\n", k, m[k])
	}
	return buf.String()
}

func testFastWalk(t *testing.T, files map[string]string, callback func(path string, typ os.FileMode) error, want map[string]os.FileMode) {
	tempdir, err := ioutil.TempDir("", "test-fast-walk")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tempdir)
	for path, contents := range files {
		file := filepath.Join(tempdir, "/src", path)
		if err := os.MkdirAll(filepath.Dir(file), 0755); err != nil {
			t.Fatal(err)
		}
		var err error
		if strings.HasPrefix(contents, "LINK:") {
			err = os.Symlink(strings.TrimPrefix(contents, "LINK:"), file)
		} else {
			err = ioutil.WriteFile(file, []byte(contents), 0644)
		}
		if err != nil {
			t.Fatal(err)
		}
	}
	got := map[string]os.FileMode{}
	var mu sync.Mutex
	if err := fastwalk.Walk(tempdir, func(path string, typ os.FileMode) error {
		mu.Lock()
		defer mu.Unlock()
		if !strings.HasPrefix(path, tempdir) {
			t.Fatalf("bogus prefix on %q, expect %q", path, tempdir)
		}
		key := filepath.ToSlash(strings.TrimPrefix(path, tempdir))
		if old, dup := got[key]; dup {
			t.Fatalf("callback called twice for key %q: %v -> %v", key, old, typ)
		}
		got[key] = typ
		return callback(path, typ)
	}); err != nil {
		t.Fatalf("callback returned: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("walk mismatch.\n got:\n%v\nwant:\n%v", formatFileModes(got), formatFileModes(want))
	}
}

func TestFastWalk_Basic(t *testing.T) {
	testFastWalk(t, map[string]string{
		"foo/foo.go":   "one",
		"bar/bar.go":   "two",
		"skip/skip.go": "skip",
	},
		func(path string, typ os.FileMode) error {
			return nil
		},
		map[string]os.FileMode{
			"":                  os.ModeDir,
			"/src":              os.ModeDir,
			"/src/bar":          os.ModeDir,
			"/src/bar/bar.go":   0,
			"/src/foo":          os.ModeDir,
			"/src/foo/foo.go":   0,
			"/src/skip":         os.ModeDir,
			"/src/skip/skip.go": 0,
		})
}

func TestFastWalk_LongFileName(t *testing.T) {
	longFileName := strings.Repeat("x", 255)

	testFastWalk(t, map[string]string{
		longFileName: "one",
	},
		func(path string, typ os.FileMode) error {
			return nil
		},
		map[string]os.FileMode{
			"":                     os.ModeDir,
			"/src":                 os.ModeDir,
			"/src/" + longFileName: 0,
		},
	)
}

func TestFastWalk_Symlink(t *testing.T) {
	switch runtime.GOOS {
	case "windows", "plan9":
		t.Skipf("skipping on %s", runtime.GOOS)
	}
	testFastWalk(t, map[string]string{
		"foo/foo.go": "one",
		"bar/bar.go": "LINK:../foo.go",
		"symdir":     "LINK:foo",
	},
		func(path string, typ os.FileMode) error {
			return nil
		},
		map[string]os.FileMode{
			"":                os.ModeDir,
			"/src":            os.ModeDir,
			"/src/bar":        os.ModeDir,
			"/src/bar/bar.go": os.ModeSymlink,
			"/src/foo":        os.ModeDir,
			"/src/foo/foo.go": 0,
			"/src/symdir":     os.ModeSymlink,
		})
}

func TestFastWalk_SkipDir(t *testing.T) {
	testFastWalk(t, map[string]string{
		"foo/foo.go":   "one",
		"bar/bar.go":   "two",
		"skip/skip.go": "skip",
	},
		func(path string, typ os.FileMode) error {
			if typ == os.ModeDir && strings.HasSuffix(path, "skip") {
				return filepath.SkipDir
			}
			return nil
		},
		map[string]os.FileMode{
			"":                os.ModeDir,
			"/src":            os.ModeDir,
			"/src/bar":        os.ModeDir,
			"/src/bar/bar.go": 0,
			"/src/foo":        os.ModeDir,
			"/src/foo/foo.go": 0,
			"/src/skip":       os.ModeDir,
		})
}

func TestFastWalk_SkipFiles(t *testing.T) {
	// Directory iteration order is undefined, so there's no way to know
	// which file to expect until the walk happens. Rather than mess
	// with the test infrastructure, just mutate want.
	var mu sync.Mutex
	want := map[string]os.FileMode{
		"":              os.ModeDir,
		"/src":          os.ModeDir,
		"/src/zzz":      os.ModeDir,
		"/src/zzz/c.go": 0,
	}

	testFastWalk(t, map[string]string{
		"a_skipfiles.go": "a",
		"b_skipfiles.go": "b",
		"zzz/c.go":       "c",
	},
		func(path string, typ os.FileMode) error {
			if strings.HasSuffix(path, "_skipfiles.go") {
				mu.Lock()
				defer mu.Unlock()
				want["/src/"+filepath.Base(path)] = 0
				return fastwalk.ErrSkipFiles
			}
			return nil
		},
		want)
	if len(want) != 5 {
		t.Errorf("saw too many files: wanted 5, got %v (%v)", len(want), want)
	}
}

func TestFastWalk_TraverseSymlink(t *testing.T) {
	switch runtime.GOOS {
	case "windows", "plan9":
		t.Skipf("skipping on %s", runtime.GOOS)
	}

	testFastWalk(t, map[string]string{
		"foo/foo.go":   "one",
		"bar/bar.go":   "two",
		"skip/skip.go": "skip",
		"symdir":       "LINK:foo",
	},
		func(path string, typ os.FileMode) error {
			if typ == os.ModeSymlink {
				return fastwalk.ErrTraverseLink
			}
			return nil
		},
		map[string]os.FileMode{
			"":                   os.ModeDir,
			"/src":               os.ModeDir,
			"/src/bar":           os.ModeDir,
			"/src/bar/bar.go":    0,
			"/src/foo":           os.ModeDir,
			"/src/foo/foo.go":    0,
			"/src/skip":          os.ModeDir,
			"/src/skip/skip.go":  0,
			"/src/symdir":        os.ModeSymlink,
			"/src/symdir/foo.go": 0,
		})
}

var benchDir = flag.String("benchdir", runtime.GOROOT(), "The directory to scan for BenchmarkFastWalk")

func BenchmarkFastWalk(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		err := fastwalk.Walk(*benchDir, func(path string, typ os.FileMode) error { return nil })
		if err != nil {
			b.Fatal(err)
		}
	}
}
