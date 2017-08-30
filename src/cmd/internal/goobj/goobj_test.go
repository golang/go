// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package goobj

import (
	"fmt"
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

var (
	buildDir  string
	go1obj    string
	go2obj    string
	goarchive string
)

func TestMain(m *testing.M) {
	if !testenv.HasGoBuild() {
		return
	}

	if runtime.GOOS == "linux" && runtime.GOARCH == "arm" {
		return // skip tests due to #19811
	}

	if err := buildGoobj(); err != nil {
		fmt.Println(err)
		os.RemoveAll(buildDir)
		os.Exit(1)
	}

	exit := m.Run()

	os.RemoveAll(buildDir)
	os.Exit(exit)
}

func buildGoobj() error {
	var err error

	buildDir, err = ioutil.TempDir("", "TestGoobj")
	if err != nil {
		return err
	}

	go1obj = filepath.Join(buildDir, "go1.o")
	go2obj = filepath.Join(buildDir, "go2.o")
	goarchive = filepath.Join(buildDir, "go.a")

	gotool, err := testenv.GoTool()
	if err != nil {
		return err
	}

	go1src := filepath.Join("testdata", "go1.go")
	go2src := filepath.Join("testdata", "go2.go")

	out, err := exec.Command(gotool, "tool", "compile", "-o", go1obj, go1src).CombinedOutput()
	if err != nil {
		return fmt.Errorf("go tool compile -o %s %s: %v\n%s", go1obj, go1src, err, out)
	}
	out, err = exec.Command(gotool, "tool", "compile", "-o", go2obj, go2src).CombinedOutput()
	if err != nil {
		return fmt.Errorf("go tool compile -o %s %s: %v\n%s", go2obj, go2src, err, out)
	}
	out, err = exec.Command(gotool, "tool", "pack", "c", goarchive, go1obj, go2obj).CombinedOutput()
	if err != nil {
		return fmt.Errorf("go tool pack c %s %s %s: %v\n%s", goarchive, go1obj, go2obj, err, out)
	}

	return nil
}

func TestParseGoobj(t *testing.T) {
	path := go1obj

	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	p, err := Parse(f, "mypkg")
	if err != nil {
		t.Fatal(err)
	}
	if p.Arch != runtime.GOARCH {
		t.Errorf("%s: got %v, want %v", path, p.Arch, runtime.GOARCH)
	}
	var found bool
	for _, s := range p.Syms {
		if s.Name == "mypkg.go1" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf(`%s: symbol "mypkg.go1" not found`, path)
	}
}

func TestParseArchive(t *testing.T) {
	path := goarchive

	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	p, err := Parse(f, "mypkg")
	if err != nil {
		t.Fatal(err)
	}
	if p.Arch != runtime.GOARCH {
		t.Errorf("%s: got %v, want %v", path, p.Arch, runtime.GOARCH)
	}
	var found1 bool
	var found2 bool
	for _, s := range p.Syms {
		if s.Name == "mypkg.go1" {
			found1 = true
		}
		if s.Name == "mypkg.go2" {
			found2 = true
		}
	}
	if !found1 {
		t.Errorf(`%s: symbol "mypkg.go1" not found`, path)
	}
	if !found2 {
		t.Errorf(`%s: symbol "mypkg.go2" not found`, path)
	}
}
