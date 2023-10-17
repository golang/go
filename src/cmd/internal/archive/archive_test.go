// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package archive

import (
	"bytes"
	"debug/elf"
	"debug/macho"
	"debug/pe"
	"fmt"
	"internal/testenv"
	"internal/xcoff"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"testing"
	"unicode/utf8"
)

var buildDir string

func TestMain(m *testing.M) {
	if !testenv.HasGoBuild() {
		return
	}

	exit := m.Run()

	if buildDir != "" {
		os.RemoveAll(buildDir)
	}
	os.Exit(exit)
}

func copyDir(dst, src string) error {
	err := os.MkdirAll(dst, 0777)
	if err != nil {
		return err
	}
	entries, err := os.ReadDir(src)
	if err != nil {
		return err
	}
	for _, entry := range entries {
		err = copyFile(filepath.Join(dst, entry.Name()), filepath.Join(src, entry.Name()))
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

var (
	buildOnce   sync.Once
	builtGoobjs goobjPaths
	buildErr    error
)

type goobjPaths struct {
	go1obj     string
	go2obj     string
	goarchive  string
	cgoarchive string
}

func buildGoobj(t *testing.T) goobjPaths {
	buildOnce.Do(func() {
		buildErr = func() (err error) {
			buildDir, err = os.MkdirTemp("", "TestGoobj")
			if err != nil {
				return err
			}

			go1obj := filepath.Join(buildDir, "go1.o")
			go2obj := filepath.Join(buildDir, "go2.o")
			goarchive := filepath.Join(buildDir, "go.a")
			cgoarchive := ""

			gotool, err := testenv.GoTool()
			if err != nil {
				return err
			}

			go1src := filepath.Join("testdata", "go1.go")
			go2src := filepath.Join("testdata", "go2.go")

			importcfgfile := filepath.Join(buildDir, "importcfg")
			testenv.WriteImportcfg(t, importcfgfile, nil, go1src, go2src)

			out, err := testenv.Command(t, gotool, "tool", "compile", "-importcfg="+importcfgfile, "-p=p", "-o", go1obj, go1src).CombinedOutput()
			if err != nil {
				return fmt.Errorf("go tool compile -o %s %s: %v\n%s", go1obj, go1src, err, out)
			}
			out, err = testenv.Command(t, gotool, "tool", "compile", "-importcfg="+importcfgfile, "-p=p", "-o", go2obj, go2src).CombinedOutput()
			if err != nil {
				return fmt.Errorf("go tool compile -o %s %s: %v\n%s", go2obj, go2src, err, out)
			}
			out, err = testenv.Command(t, gotool, "tool", "pack", "c", goarchive, go1obj, go2obj).CombinedOutput()
			if err != nil {
				return fmt.Errorf("go tool pack c %s %s %s: %v\n%s", goarchive, go1obj, go2obj, err, out)
			}

			if testenv.HasCGO() {
				cgoarchive = filepath.Join(buildDir, "mycgo.a")
				gopath := filepath.Join(buildDir, "gopath")
				err = copyDir(filepath.Join(gopath, "src", "mycgo"), filepath.Join("testdata", "mycgo"))
				if err == nil {
					err = os.WriteFile(filepath.Join(gopath, "src", "mycgo", "go.mod"), []byte("module mycgo\n"), 0666)
				}
				if err != nil {
					return err
				}
				cmd := testenv.Command(t, gotool, "build", "-buildmode=archive", "-o", cgoarchive, "-gcflags=all="+os.Getenv("GO_GCFLAGS"), "mycgo")
				cmd.Dir = filepath.Join(gopath, "src", "mycgo")
				cmd.Env = append(os.Environ(), "GOPATH="+gopath)
				out, err = cmd.CombinedOutput()
				if err != nil {
					return fmt.Errorf("go install mycgo: %v\n%s", err, out)
				}
			}

			builtGoobjs = goobjPaths{
				go1obj:     go1obj,
				go2obj:     go2obj,
				goarchive:  goarchive,
				cgoarchive: cgoarchive,
			}
			return nil
		}()
	})

	if buildErr != nil {
		t.Helper()
		t.Fatal(buildErr)
	}
	return builtGoobjs
}

func TestParseGoobj(t *testing.T) {
	path := buildGoobj(t).go1obj

	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	a, err := Parse(f, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(a.Entries) != 2 {
		t.Errorf("expect 2 entry, found %d", len(a.Entries))
	}
	for _, e := range a.Entries {
		if e.Type == EntryPkgDef {
			continue
		}
		if e.Type != EntryGoObj {
			t.Errorf("wrong type of object: want EntryGoObj, got %v", e.Type)
		}
		if !bytes.Contains(e.Obj.TextHeader, []byte(runtime.GOARCH)) {
			t.Errorf("text header does not contain GOARCH %s: %q", runtime.GOARCH, e.Obj.TextHeader)
		}
	}
}

func TestParseArchive(t *testing.T) {
	path := buildGoobj(t).goarchive

	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	a, err := Parse(f, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(a.Entries) != 3 {
		t.Errorf("expect 3 entry, found %d", len(a.Entries))
	}
	var found1 bool
	var found2 bool
	for _, e := range a.Entries {
		if e.Type == EntryPkgDef {
			continue
		}
		if e.Type != EntryGoObj {
			t.Errorf("wrong type of object: want EntryGoObj, got %v", e.Type)
		}
		if !bytes.Contains(e.Obj.TextHeader, []byte(runtime.GOARCH)) {
			t.Errorf("text header does not contain GOARCH %s: %q", runtime.GOARCH, e.Obj.TextHeader)
		}
		if e.Name == "go1.o" {
			found1 = true
		}
		if e.Name == "go2.o" {
			found2 = true
		}
	}
	if !found1 {
		t.Errorf(`object "go1.o" not found`)
	}
	if !found2 {
		t.Errorf(`object "go2.o" not found`)
	}
}

func TestParseCGOArchive(t *testing.T) {
	testenv.MustHaveCGO(t)

	path := buildGoobj(t).cgoarchive

	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	a, err := Parse(f, false)
	if err != nil {
		t.Fatal(err)
	}

	c1 := "c1"
	c2 := "c2"
	switch runtime.GOOS {
	case "darwin", "ios":
		c1 = "_" + c1
		c2 = "_" + c2
	case "windows":
		if runtime.GOARCH == "386" {
			c1 = "_" + c1
			c2 = "_" + c2
		}
	case "aix":
		c1 = "." + c1
		c2 = "." + c2
	}

	var foundgo, found1, found2 bool

	for _, e := range a.Entries {
		switch e.Type {
		default:
			t.Errorf("unknown object type")
		case EntryPkgDef:
			continue
		case EntryGoObj:
			foundgo = true
			if !bytes.Contains(e.Obj.TextHeader, []byte(runtime.GOARCH)) {
				t.Errorf("text header does not contain GOARCH %s: %q", runtime.GOARCH, e.Obj.TextHeader)
			}
			continue
		case EntryNativeObj:
		}

		obj := io.NewSectionReader(f, e.Offset, e.Size)
		switch runtime.GOOS {
		case "darwin", "ios":
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
		case "windows":
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
		case "aix":
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
		default: // ELF
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

	if !foundgo {
		t.Errorf(`go object not found`)
	}
	if !found1 {
		t.Errorf(`symbol %q not found`, c1)
	}
	if !found2 {
		t.Errorf(`symbol %q not found`, c2)
	}
}

func TestExactly16Bytes(t *testing.T) {
	var tests = []string{
		"",
		"a",
		"日本語",
		"1234567890123456",
		"12345678901234567890",
		"1234567890123本語4567890",
		"12345678901234日本語567890",
		"123456789012345日本語67890",
		"1234567890123456日本語7890",
		"1234567890123456日本語7日本語890",
	}
	for _, str := range tests {
		got := exactly16Bytes(str)
		if len(got) != 16 {
			t.Errorf("exactly16Bytes(%q) is %q, length %d", str, got, len(got))
		}
		// Make sure it is full runes.
		for _, c := range got {
			if c == utf8.RuneError {
				t.Errorf("exactly16Bytes(%q) is %q, has partial rune", str, got)
			}
		}
	}
}
