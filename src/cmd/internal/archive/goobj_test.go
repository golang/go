// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package goobj

import (
	"debug/elf"
	"debug/macho"
	"debug/pe"
	"fmt"
	"internal/testenv"
	"internal/xcoff"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

var (
	buildDir   string
	go1obj     string
	go2obj     string
	goarchive  string
	cgoarchive string
)

func TestMain(m *testing.M) {
	if !testenv.HasGoBuild() {
		return
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

func copyDir(dst, src string) error {
	err := os.MkdirAll(dst, 0777)
	if err != nil {
		return err
	}
	fis, err := ioutil.ReadDir(src)
	if err != nil {
		return err
	}
	for _, fi := range fis {
		err = copyFile(filepath.Join(dst, fi.Name()), filepath.Join(src, fi.Name()))
		if err != nil {
			return err
		}
	}
	return nil
}

func copyFile(dst, src string) (err error) {
	var s, d *os.File
	s, err = os.Open(src)
	if err != nil {
		return err
	}
	defer s.Close()
	d, err = os.Create(dst)
	if err != nil {
		return err
	}
	defer func() {
		e := d.Close()
		if err == nil {
			err = e
		}
	}()
	_, err = io.Copy(d, s)
	if err != nil {
		return err
	}
	return nil
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

	if testenv.HasCGO() {
		gopath := filepath.Join(buildDir, "gopath")
		err = copyDir(filepath.Join(gopath, "src", "mycgo"), filepath.Join("testdata", "mycgo"))
		if err == nil {
			err = ioutil.WriteFile(filepath.Join(gopath, "src", "mycgo", "go.mod"), []byte("module mycgo\n"), 0666)
		}
		if err != nil {
			return err
		}
		cmd := exec.Command(gotool, "install", "-gcflags=all="+os.Getenv("GO_GCFLAGS"), "mycgo")
		cmd.Dir = filepath.Join(gopath, "src", "mycgo")
		cmd.Env = append(os.Environ(), "GOPATH="+gopath)
		out, err = cmd.CombinedOutput()
		if err != nil {
			return fmt.Errorf("go install mycgo: %v\n%s", err, out)
		}
		pat := filepath.Join(gopath, "pkg", "*", "mycgo.a")
		ms, err := filepath.Glob(pat)
		if err != nil {
			return err
		}
		if len(ms) == 0 {
			return fmt.Errorf("cannot found paths for pattern %s", pat)
		}
		cgoarchive = ms[0]
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

func TestParseCGOArchive(t *testing.T) {
	testenv.MustHaveCGO(t)

	path := cgoarchive

	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	p, err := Parse(f, "mycgo")
	if err != nil {
		t.Fatal(err)
	}
	if p.Arch != runtime.GOARCH {
		t.Errorf("%s: got %v, want %v", path, p.Arch, runtime.GOARCH)
	}
	var found1 bool
	var found2 bool
	for _, s := range p.Syms {
		if s.Name == "mycgo.go1" {
			found1 = true
		}
		if s.Name == "mycgo.go2" {
			found2 = true
		}
	}
	if !found1 {
		t.Errorf(`%s: symbol "mycgo.go1" not found`, path)
	}
	if !found2 {
		t.Errorf(`%s: symbol "mycgo.go2" not found`, path)
	}

	c1 := "c1"
	c2 := "c2"

	found1 = false
	found2 = false

	switch runtime.GOOS {
	case "darwin":
		c1 = "_" + c1
		c2 = "_" + c2
		for _, obj := range p.Native {
			mf, err := macho.NewFile(obj)
			if err != nil {
				t.Fatal(err)
			}
			if mf.Symtab == nil {
				continue
			}
			for _, s := range mf.Symtab.Syms {
				switch s.Name {
				case c1:
					found1 = true
				case c2:
					found2 = true
				}
			}
		}
	case "windows":
		if runtime.GOARCH == "386" {
			c1 = "_" + c1
			c2 = "_" + c2
		}
		for _, obj := range p.Native {
			pf, err := pe.NewFile(obj)
			if err != nil {
				t.Fatal(err)
			}
			for _, s := range pf.Symbols {
				switch s.Name {
				case c1:
					found1 = true
				case c2:
					found2 = true
				}
			}
		}
	case "aix":
		c1 = "." + c1
		c2 = "." + c2
		for _, obj := range p.Native {
			xf, err := xcoff.NewFile(obj)
			if err != nil {
				t.Fatal(err)
			}
			for _, s := range xf.Symbols {
				switch s.Name {
				case c1:
					found1 = true
				case c2:
					found2 = true
				}
			}
		}

	default:
		for _, obj := range p.Native {
			ef, err := elf.NewFile(obj)
			if err != nil {
				t.Fatal(err)
			}
			syms, err := ef.Symbols()
			if err != nil {
				t.Fatal(err)
			}
			for _, s := range syms {
				switch s.Name {
				case c1:
					found1 = true
				case c2:
					found2 = true
				}
			}
		}
	}

	if !found1 {
		t.Errorf(`%s: symbol %q not found`, path, c1)
	}
	if !found2 {
		t.Errorf(`%s: symbol %q not found`, path, c2)
	}
}
