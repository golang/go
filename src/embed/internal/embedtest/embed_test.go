// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package embedtest

import (
	"embed"
	"reflect"
	"testing"
	"testing/fstest"
)

//go:embed testdata/h*.txt
//go:embed c*.txt testdata/g*.txt
var global embed.FS

//go:embed c*txt
var concurrency string

//go:embed testdata/g*.txt
var glass []byte

func testFiles(t *testing.T, f embed.FS, name, data string) {
	t.Helper()
	d, err := f.ReadFile(name)
	if err != nil {
		t.Error(err)
		return
	}
	if string(d) != data {
		t.Errorf("read %v = %q, want %q", name, d, data)
	}
}

func testString(t *testing.T, s, name, data string) {
	t.Helper()
	if s != data {
		t.Errorf("%v = %q, want %q", name, s, data)
	}
}

func testDir(t *testing.T, f embed.FS, name string, expect ...string) {
	t.Helper()
	dirs, err := f.ReadDir(name)
	if err != nil {
		t.Error(err)
		return
	}
	var names []string
	for _, d := range dirs {
		name := d.Name()
		if d.IsDir() {
			name += "/"
		}
		names = append(names, name)
	}
	if !reflect.DeepEqual(names, expect) {
		t.Errorf("readdir %v = %v, want %v", name, names, expect)
	}
}

func TestGlobal(t *testing.T) {
	testFiles(t, global, "concurrency.txt", "Concurrency is not parallelism.\n")
	testFiles(t, global, "testdata/hello.txt", "hello, world\n")
	testFiles(t, global, "testdata/glass.txt", "I can eat glass and it doesn't hurt me.\n")

	if err := fstest.TestFS(global); err != nil {
		t.Fatal(err)
	}

	testString(t, concurrency, "concurrency", "Concurrency is not parallelism.\n")
	testString(t, string(glass), "glass", "I can eat glass and it doesn't hurt me.\n")
}

func TestLocal(t *testing.T) {
	//go:embed testdata/k*.txt
	var local embed.FS
	testFiles(t, local, "testdata/ken.txt", "If a program is too slow, it must have a loop.\n")

	//go:embed testdata/k*.txt
	var s string
	testString(t, s, "local variable s", "If a program is too slow, it must have a loop.\n")

	//go:embed testdata/h*.txt
	var b []byte
	testString(t, string(b), "local variable b", "hello, world\n")
}

func TestDir(t *testing.T) {
	//go:embed testdata
	var all embed.FS

	testFiles(t, all, "testdata/hello.txt", "hello, world\n")
	testFiles(t, all, "testdata/i/i18n.txt", "internationalization\n")
	testFiles(t, all, "testdata/i/j/k/k8s.txt", "kubernetes\n")
	testFiles(t, all, "testdata/ken.txt", "If a program is too slow, it must have a loop.\n")

	testDir(t, all, ".", "testdata/")
	testDir(t, all, "testdata/i", "i18n.txt", "j/")
	testDir(t, all, "testdata/i/j", "k/")
	testDir(t, all, "testdata/i/j/k", "k8s.txt")
}
