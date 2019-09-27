// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vfs_test

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"golang.org/x/tools/godoc/vfs"
	"golang.org/x/tools/godoc/vfs/mapfs"
)

func TestNewNameSpace(t *testing.T) {

	// We will mount this filesystem under /fs1
	mount := mapfs.New(map[string]string{"fs1file": "abcdefgh"})

	// Existing process. This should give error on Stat("/")
	t1 := vfs.NameSpace{}
	t1.Bind("/fs1", mount, "/", vfs.BindReplace)

	// using NewNameSpace. This should work fine.
	t2 := vfs.NewNameSpace()
	t2.Bind("/fs1", mount, "/", vfs.BindReplace)

	testcases := map[string][]bool{
		"/":            {false, true},
		"/fs1":         {true, true},
		"/fs1/fs1file": {true, true},
	}

	fss := []vfs.FileSystem{t1, t2}

	for j, fs := range fss {
		for k, v := range testcases {
			_, err := fs.Stat(k)
			result := err == nil
			if result != v[j] {
				t.Errorf("fs: %d, testcase: %s, want: %v, got: %v, err: %s", j, k, v[j], result, err)
			}
		}
	}

	fi, err := t2.Stat("/")
	if err != nil {
		t.Fatal(err)
	}

	if fi.Name() != "/" {
		t.Errorf("t2.Name() : want:%s got:%s", "/", fi.Name())
	}

	if !fi.ModTime().IsZero() {
		t.Errorf("t2.ModTime() : want:%v got:%v", time.Time{}, fi.ModTime())
	}
}

func TestReadDirUnion(t *testing.T) {
	for _, tc := range []struct {
		desc       string
		ns         vfs.NameSpace
		path, want string
	}{
		{
			desc: "no_go_files",
			ns: func() vfs.NameSpace {
				rootFs := mapfs.New(map[string]string{
					"doc/a.txt":       "1",
					"doc/b.txt":       "1",
					"doc/dir1/d1.txt": "",
				})
				docFs := mapfs.New(map[string]string{
					"doc/a.txt":       "22",
					"doc/dir2/d2.txt": "",
				})
				ns := vfs.NameSpace{}
				ns.Bind("/", rootFs, "/", vfs.BindReplace)
				ns.Bind("/doc", docFs, "/doc", vfs.BindBefore)
				return ns
			}(),
			path: "/doc",
			want: "a.txt:2,b.txt:1,dir1:0,dir2:0",
		}, {
			desc: "have_go_files",
			ns: func() vfs.NameSpace {
				a := mapfs.New(map[string]string{
					"src/x/a.txt":        "",
					"src/x/suba/sub.txt": "",
				})
				b := mapfs.New(map[string]string{
					"src/x/b.go":         "package b",
					"src/x/subb/sub.txt": "",
				})
				c := mapfs.New(map[string]string{
					"src/x/c.txt":        "",
					"src/x/subc/sub.txt": "",
				})
				ns := vfs.NameSpace{}
				ns.Bind("/", a, "/", vfs.BindReplace)
				ns.Bind("/", b, "/", vfs.BindAfter)
				ns.Bind("/", c, "/", vfs.BindAfter)
				return ns
			}(),
			path: "/src/x",
			want: "b.go:9,suba:0,subb:0,subc:0",
		}, {
			desc: "empty_mount",
			ns: func() vfs.NameSpace {
				ns := vfs.NameSpace{}
				ns.Bind("/empty", mapfs.New(nil), "/empty", vfs.BindReplace)
				return ns
			}(),
			path: "/",
			want: "empty:0",
		},
	} {
		t.Run(tc.desc, func(t *testing.T) {
			fis, err := tc.ns.ReadDir(tc.path)
			if err != nil {
				t.Fatal(err)
			}
			buf := &strings.Builder{}
			sep := ""
			for _, fi := range fis {
				fmt.Fprintf(buf, "%s%s:%d", sep, fi.Name(), fi.Size())
				sep = ","
			}
			if got := buf.String(); got != tc.want {
				t.Errorf("got %q; want %q", got, tc.want)
			}
		})
	}
}
