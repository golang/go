// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vfs

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

type fakeFileInfo struct {
	dir      bool
	link     bool
	basename string
	modtime  time.Time
	ents     []*fakeFileInfo
	contents string
	err      error
}

func (f *fakeFileInfo) Name() string       { return f.basename }
func (f *fakeFileInfo) Sys() interface{}   { return nil }
func (f *fakeFileInfo) ModTime() time.Time { return f.modtime }
func (f *fakeFileInfo) IsDir() bool        { return f.dir }
func (f *fakeFileInfo) Size() int64        { return int64(len(f.contents)) }
func (f *fakeFileInfo) Mode() os.FileMode {
	if f.dir {
		return 0755 | os.ModeDir
	}
	if f.link {
		return 0644 | os.ModeSymlink
	}
	return 0644
}

func TestOSReadDirFollowsSymLinks(t *testing.T) {
	// Stat is called on ReadDir output by osFS
	oldstat := stat
	stat = func(path string) (os.FileInfo, error) {
		if filepath.ToSlash(path) != "/tmp/subdir/is_link" {
			t.Fatalf("stat called on unexpected path %q", path)
		}
		return &fakeFileInfo{
			link:     false,
			basename: "foo",
		}, nil
	}
	defer func() { stat = oldstat }()

	oldreaddir := readdir
	readdir = func(path string) ([]os.FileInfo, error) {
		return []os.FileInfo{
			&fakeFileInfo{
				link:     true,
				basename: "is_link",
			},
			&fakeFileInfo{
				link:     false,
				basename: "not_link",
			},
		}, nil
	}
	defer func() { readdir = oldreaddir }()

	fs := OS("/tmp")
	result, err := fs.ReadDir("subdir")

	if err != nil {
		t.Fatal(err)
	}

	var gotBuf bytes.Buffer
	for i, fi := range result {
		fmt.Fprintf(&gotBuf, "result[%d] = %v, %v\n", i, fi.Name(), fi.Mode())
	}
	got := gotBuf.String()
	want := `result[0] = foo, -rw-r--r--
result[1] = not_link, -rw-r--r--
`
	if got != want {
		t.Errorf("ReadDir got:\n%s\n\nwant:\n%s\n", got, want)
	}
}

func TestOSReadDirHandlesReadDirErrors(t *testing.T) {
	oldreaddir := readdir
	readdir = func(path string) ([]os.FileInfo, error) {
		return []os.FileInfo{
			&fakeFileInfo{
				dir:      false,
				basename: "foo",
			},
		}, errors.New("some arbitrary filesystem failure")
	}
	defer func() { readdir = oldreaddir }()

	fs := OS("/tmp")
	_, err := fs.ReadDir("subdir")

	if got, want := fmt.Sprint(err), "some arbitrary filesystem failure"; got != want {
		t.Errorf("ReadDir = %v; want %q", got, want)
	}
}
