// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.package zipfs
package zipfs

import (
	"archive/zip"
	"bytes"
	"fmt"
	"io"
	"os"
	"reflect"
	"testing"

	"golang.org/x/tools/godoc/vfs"
)

var (

	// files to use to build zip used by zipfs in testing; maps path : contents
	files = map[string]string{"foo": "foo", "bar/baz": "baz", "a/b/c": "c"}

	// expected info for each entry in a file system described by files
	tests = []struct {
		Path      string
		IsDir     bool
		IsRegular bool
		Name      string
		Contents  string
		Files     map[string]bool
	}{
		{"/", true, false, "", "", map[string]bool{"foo": true, "bar": true, "a": true}},
		{"//", true, false, "", "", map[string]bool{"foo": true, "bar": true, "a": true}},
		{"/foo", false, true, "foo", "foo", nil},
		{"/foo/", false, true, "foo", "foo", nil},
		{"/foo//", false, true, "foo", "foo", nil},
		{"/bar", true, false, "bar", "", map[string]bool{"baz": true}},
		{"/bar/", true, false, "bar", "", map[string]bool{"baz": true}},
		{"/bar/baz", false, true, "baz", "baz", nil},
		{"//bar//baz", false, true, "baz", "baz", nil},
		{"/a/b", true, false, "b", "", map[string]bool{"c": true}},
	}

	// to be initialized in setup()
	fs        vfs.FileSystem
	statFuncs []statFunc
)

type statFunc struct {
	Name string
	Func func(string) (os.FileInfo, error)
}

func TestMain(t *testing.M) {
	if err := setup(); err != nil {
		fmt.Fprintf(os.Stderr, "Error setting up zipfs testing state: %v.\n", err)
		os.Exit(1)
	}
	os.Exit(t.Run())
}

// setups state each of the tests uses
func setup() error {
	// create zipfs
	b := new(bytes.Buffer)
	zw := zip.NewWriter(b)
	for file, contents := range files {
		w, err := zw.Create(file)
		if err != nil {
			return err
		}
		_, err = io.WriteString(w, contents)
		if err != nil {
			return err
		}
	}
	zw.Close()
	zr, err := zip.NewReader(bytes.NewReader(b.Bytes()), int64(b.Len()))
	if err != nil {
		return err
	}
	rc := &zip.ReadCloser{
		Reader: *zr,
	}
	fs = New(rc, "foo")

	// pull out different stat functions
	statFuncs = []statFunc{
		{"Stat", fs.Stat},
		{"Lstat", fs.Lstat},
	}

	return nil
}

func TestZipFSReadDir(t *testing.T) {
	for _, test := range tests {
		if test.IsDir {
			infos, err := fs.ReadDir(test.Path)
			if err != nil {
				t.Errorf("Failed to read directory %v\n", test.Path)
				continue
			}
			got := make(map[string]bool)
			for _, info := range infos {
				got[info.Name()] = true
			}
			if want := test.Files; !reflect.DeepEqual(got, want) {
				t.Errorf("ReadDir %v got %v\nwanted %v\n", test.Path, got, want)
			}
		}
	}
}

func TestZipFSStatFuncs(t *testing.T) {
	for _, test := range tests {
		for _, statFunc := range statFuncs {

			// test can stat
			info, err := statFunc.Func(test.Path)
			if err != nil {
				t.Errorf("Unexpected error using %v for %v: %v\n", statFunc.Name, test.Path, err)
				continue
			}

			// test info.Name()
			if got, want := info.Name(), test.Name; got != want {
				t.Errorf("Using %v for %v info.Name() got %v wanted %v\n", statFunc.Name, test.Path, got, want)
			}
			// test info.IsDir()
			if got, want := info.IsDir(), test.IsDir; got != want {
				t.Errorf("Using %v for %v info.IsDir() got %v wanted %v\n", statFunc.Name, test.Path, got, want)
			}
			// test info.Mode().IsDir()
			if got, want := info.Mode().IsDir(), test.IsDir; got != want {
				t.Errorf("Using %v for %v info.Mode().IsDir() got %v wanted %v\n", statFunc.Name, test.Path, got, want)
			}
			// test info.Mode().IsRegular()
			if got, want := info.Mode().IsRegular(), test.IsRegular; got != want {
				t.Errorf("Using %v for %v info.Mode().IsRegular() got %v wanted %v\n", statFunc.Name, test.Path, got, want)
			}
			// test info.Size()
			if test.IsRegular {
				if got, want := info.Size(), int64(len(test.Contents)); got != want {
					t.Errorf("Using %v for %v inf.Size() got %v wanted %v", statFunc.Name, test.Path, got, want)
				}
			}
		}
	}
}

func TestZipFSNotExist(t *testing.T) {
	_, err := fs.Open("/does-not-exist")
	if err == nil {
		t.Fatalf("Expected an error.\n")
	}
	if !os.IsNotExist(err) {
		t.Errorf("Expected an error satisfying os.IsNotExist: %v\n", err)
	}
}

func TestZipFSOpenSeek(t *testing.T) {
	for _, test := range tests {
		if test.IsRegular {

			// test Open()
			f, err := fs.Open(test.Path)
			if err != nil {
				t.Error(err)
				return
			}
			defer f.Close()

			// test Seek() multiple times
			for i := 0; i < 3; i++ {
				all, err := io.ReadAll(f)
				if err != nil {
					t.Error(err)
					return
				}
				if got, want := string(all), test.Contents; got != want {
					t.Errorf("File contents for %v got %v wanted %v\n", test.Path, got, want)
				}
				f.Seek(0, 0)
			}
		}
	}
}

func TestRootType(t *testing.T) {
	tests := []struct {
		path   string
		fsType vfs.RootType
	}{
		{"/src/net/http", vfs.RootTypeGoRoot},
		{"/src/badpath", ""},
		{"/", vfs.RootTypeGoRoot},
	}

	for _, item := range tests {
		if fs.RootType(item.path) != item.fsType {
			t.Errorf("unexpected fsType. Expected- %v, Got- %v", item.fsType, fs.RootType(item.path))
		}
	}
}
