// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testcache

import (
	"io/ioutil"
	"os"
	"runtime"
	"testing"
)

func TestChdir(t *testing.T) {
	os.Chdir("..")
	defer os.Chdir("testcache")
	info, err := os.Stat("testcache/file.txt")
	if err != nil {
		t.Fatal(err)
	}
	if info.Size()%2 != 1 {
		t.Fatal("even file")
	}
}

func TestOddFileContent(t *testing.T) {
	f, err := os.Open("file.txt")
	if err != nil {
		t.Fatal(err)
	}
	data, err := ioutil.ReadAll(f)
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	if len(data)%2 != 1 {
		t.Fatal("even file")
	}
}

func TestOddFileSize(t *testing.T) {
	info, err := os.Stat("file.txt")
	if err != nil {
		t.Fatal(err)
	}
	if info.Size()%2 != 1 {
		t.Fatal("even file")
	}
}

func TestOddGetenv(t *testing.T) {
	val := os.Getenv("TESTKEY")
	if len(val)%2 != 1 {
		t.Fatal("even env value")
	}
}

func TestLookupEnv(t *testing.T) {
	_, ok := os.LookupEnv("TESTKEY")
	if !ok {
		t.Fatal("env missing")
	}
}

func TestDirList(t *testing.T) {
	f, err := os.Open(".")
	if err != nil {
		t.Fatal(err)
	}
	f.Readdirnames(-1)
	f.Close()
}

func TestExec(t *testing.T) {
	if runtime.GOOS == "plan9" || runtime.GOOS == "windows" || runtime.GOOS == "nacl" {
		t.Skip("non-unix")
	}

	// Note: not using os/exec to make sure there is no unexpected stat.
	p, err := os.StartProcess("./script.sh", []string{"script"}, new(os.ProcAttr))
	if err != nil {
		t.Fatal(err)
	}
	ps, err := p.Wait()
	if err != nil {
		t.Fatal(err)
	}
	if !ps.Success() {
		t.Fatalf("script failed: %v", err)
	}
}
