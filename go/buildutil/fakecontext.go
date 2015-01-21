package buildutil

import (
	"fmt"
	"go/build"
	"io"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// FakeContext returns a build.Context for the fake file tree specified
// by pkgs, which maps package import paths to a mapping from file base
// names to contents.
//
// The fake Context has a GOROOT of "/go" and no GOPATH, and overrides
// the necessary file access methods to read from memory instead of the
// real file system.
//
// Unlike a real file tree, the fake one has only two levels---packages
// and files---so ReadDir("/go/src/") returns all packages under
// /go/src/ including, for instance, "math" and "math/big".
// ReadDir("/go/src/math/big") would return all the files in the
// "math/big" package.
//
func FakeContext(pkgs map[string]map[string]string) *build.Context {
	clean := func(filename string) string {
		f := path.Clean(filepath.ToSlash(filename))
		// Removing "/go/src" while respecting segment
		// boundaries has this unfortunate corner case:
		if f == "/go/src" {
			return ""
		}
		return strings.TrimPrefix(f, "/go/src/")
	}

	ctxt := build.Default // copy
	ctxt.GOROOT = "/go"
	ctxt.GOPATH = ""
	ctxt.IsDir = func(dir string) bool {
		dir = clean(dir)
		if dir == "" {
			return true // needed by (*build.Context).SrcDirs
		}
		return pkgs[dir] != nil
	}
	ctxt.ReadDir = func(dir string) ([]os.FileInfo, error) {
		dir = clean(dir)
		var fis []os.FileInfo
		if dir == "" {
			// enumerate packages
			for importPath := range pkgs {
				fis = append(fis, fakeDirInfo(importPath))
			}
		} else {
			// enumerate files of package
			for basename := range pkgs[dir] {
				fis = append(fis, fakeFileInfo(basename))
			}
		}
		sort.Sort(byName(fis))
		return fis, nil
	}
	ctxt.OpenFile = func(filename string) (io.ReadCloser, error) {
		filename = clean(filename)
		dir, base := path.Split(filename)
		content, ok := pkgs[path.Clean(dir)][base]
		if !ok {
			return nil, fmt.Errorf("file not found: %s", filename)
		}
		return ioutil.NopCloser(strings.NewReader(content)), nil
	}
	ctxt.IsAbsPath = func(path string) bool {
		path = filepath.ToSlash(path)
		// Don't rely on the default (filepath.Path) since on
		// Windows, it reports virtual paths as non-absolute.
		return strings.HasPrefix(path, "/")
	}
	return &ctxt
}

type byName []os.FileInfo

func (s byName) Len() int           { return len(s) }
func (s byName) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }
func (s byName) Less(i, j int) bool { return s[i].Name() < s[j].Name() }

type fakeFileInfo string

func (fi fakeFileInfo) Name() string    { return string(fi) }
func (fakeFileInfo) Sys() interface{}   { return nil }
func (fakeFileInfo) ModTime() time.Time { return time.Time{} }
func (fakeFileInfo) IsDir() bool        { return false }
func (fakeFileInfo) Size() int64        { return 0 }
func (fakeFileInfo) Mode() os.FileMode  { return 0644 }

type fakeDirInfo string

func (fd fakeDirInfo) Name() string    { return string(fd) }
func (fakeDirInfo) Sys() interface{}   { return nil }
func (fakeDirInfo) ModTime() time.Time { return time.Time{} }
func (fakeDirInfo) IsDir() bool        { return true }
func (fakeDirInfo) Size() int64        { return 0 }
func (fakeDirInfo) Mode() os.FileMode  { return 0755 }
