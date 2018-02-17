// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package godoc

import (
	"go/build"
	"path/filepath"
	"runtime"
	"sort"
	"testing"

	"golang.org/x/tools/godoc/vfs"
	"golang.org/x/tools/godoc/vfs/gatefs"
)

func TestNewDirTree(t *testing.T) {
	fsGate := make(chan bool, 20)
	rootfs := gatefs.New(vfs.OS(runtime.GOROOT()), fsGate)
	fs := vfs.NameSpace{}
	fs.Bind("/", rootfs, "/", vfs.BindReplace)

	c := NewCorpus(fs)
	// 3 levels deep is enough for testing
	dir := c.newDirectory("/", 3)

	processDir(t, dir)
}

func processDir(t *testing.T, dir *Directory) {
	var list []string
	for _, d := range dir.Dirs {
		list = append(list, d.Name)
		// recursively process the lower level
		processDir(t, d)
	}

	if sort.StringsAreSorted(list) == false {
		t.Errorf("list: %v is not sorted\n", list)
	}
}

func BenchmarkNewDirectory(b *testing.B) {
	if testing.Short() {
		b.Skip("not running tests requiring large file scan in short mode")
	}

	fsGate := make(chan bool, 20)

	goroot := runtime.GOROOT()
	rootfs := gatefs.New(vfs.OS(goroot), fsGate)
	fs := vfs.NameSpace{}
	fs.Bind("/", rootfs, "/", vfs.BindReplace)
	for _, p := range filepath.SplitList(build.Default.GOPATH) {
		fs.Bind("/src/golang.org", gatefs.New(vfs.OS(p), fsGate), "/src/golang.org", vfs.BindAfter)
	}
	b.ResetTimer()
	b.ReportAllocs()
	for tries := 0; tries < b.N; tries++ {
		corpus := NewCorpus(fs)
		corpus.newDirectory("/", -1)
	}
}
