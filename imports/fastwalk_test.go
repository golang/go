// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package imports

import (
	"bytes"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"sync"
	"testing"
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
	testConfig{
		gopathFiles: files,
	}.test(t, func(t *goimportTest) {
		got := map[string]os.FileMode{}
		var mu sync.Mutex
		if err := fastWalk(t.gopath, func(path string, typ os.FileMode) error {
			mu.Lock()
			defer mu.Unlock()
			if !strings.HasPrefix(path, t.gopath) {
				t.Fatal("bogus prefix on %q, expect %q", path, t.gopath)
			}
			key := filepath.ToSlash(strings.TrimPrefix(path, t.gopath))
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
	})
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
				return traverseLink
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
		err := fastWalk(*benchDir, func(path string, typ os.FileMode) error { return nil })
		if err != nil {
			b.Fatal(err)
		}
	}
}
