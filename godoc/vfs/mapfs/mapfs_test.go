// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mapfs

import (
	"io"
	"os"
	"reflect"
	"testing"
)

func TestOpenRoot(t *testing.T) {
	fs := New(map[string]string{
		"foo/bar/three.txt": "a",
		"foo/bar.txt":       "b",
		"top.txt":           "c",
		"other-top.txt":     "d",
	})
	tests := []struct {
		path string
		want string
	}{
		{"/foo/bar/three.txt", "a"},
		{"foo/bar/three.txt", "a"},
		{"foo/bar.txt", "b"},
		{"top.txt", "c"},
		{"/top.txt", "c"},
		{"other-top.txt", "d"},
		{"/other-top.txt", "d"},
	}
	for _, tt := range tests {
		rsc, err := fs.Open(tt.path)
		if err != nil {
			t.Errorf("Open(%q) = %v", tt.path, err)
			continue
		}
		slurp, err := io.ReadAll(rsc)
		if err != nil {
			t.Error(err)
		}
		if string(slurp) != tt.want {
			t.Errorf("Read(%q) = %q; want %q", tt.path, tt.want, slurp)
		}
		rsc.Close()
	}

	_, err := fs.Open("/xxxx")
	if !os.IsNotExist(err) {
		t.Errorf("ReadDir /xxxx = %v; want os.IsNotExist error", err)
	}
}

func TestReaddir(t *testing.T) {
	fs := New(map[string]string{
		"foo/bar/three.txt": "333",
		"foo/bar.txt":       "22",
		"top.txt":           "top.txt file",
		"other-top.txt":     "other-top.txt file",
	})
	tests := []struct {
		dir  string
		want []os.FileInfo
	}{
		{
			dir: "/",
			want: []os.FileInfo{
				mapFI{name: "foo", dir: true},
				mapFI{name: "other-top.txt", size: len("other-top.txt file")},
				mapFI{name: "top.txt", size: len("top.txt file")},
			},
		},
		{
			dir: "/foo",
			want: []os.FileInfo{
				mapFI{name: "bar", dir: true},
				mapFI{name: "bar.txt", size: 2},
			},
		},
		{
			dir: "/foo/",
			want: []os.FileInfo{
				mapFI{name: "bar", dir: true},
				mapFI{name: "bar.txt", size: 2},
			},
		},
		{
			dir: "/foo/bar",
			want: []os.FileInfo{
				mapFI{name: "three.txt", size: 3},
			},
		},
	}
	for _, tt := range tests {
		fis, err := fs.ReadDir(tt.dir)
		if err != nil {
			t.Errorf("ReadDir(%q) = %v", tt.dir, err)
			continue
		}
		if !reflect.DeepEqual(fis, tt.want) {
			t.Errorf("ReadDir(%q) = %#v; want %#v", tt.dir, fis, tt.want)
			continue
		}
	}

	_, err := fs.ReadDir("/xxxx")
	if !os.IsNotExist(err) {
		t.Errorf("ReadDir /xxxx = %v; want os.IsNotExist error", err)
	}
}
