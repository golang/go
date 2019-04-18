package apidiff

import (
	"bufio"
	"fmt"
	"go/types"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"golang.org/x/tools/go/packages"
)

func TestChanges(t *testing.T) {
	dir, err := ioutil.TempDir("", "apidiff_test")
	if err != nil {
		t.Fatal(err)
	}
	dir = filepath.Join(dir, "go")
	wanti, wantc := splitIntoPackages(t, dir)
	defer os.RemoveAll(dir)
	sort.Strings(wanti)
	sort.Strings(wantc)

	oldpkg, err := load("apidiff/old", dir)
	if err != nil {
		t.Fatal(err)
	}
	newpkg, err := load("apidiff/new", dir)
	if err != nil {
		t.Fatal(err)
	}

	report := Changes(oldpkg.Types, newpkg.Types)

	got := report.messages(false)
	if !reflect.DeepEqual(got, wanti) {
		t.Errorf("incompatibles: got %v\nwant %v\n", got, wanti)
	}
	got = report.messages(true)
	if !reflect.DeepEqual(got, wantc) {
		t.Errorf("compatibles: got %v\nwant %v\n", got, wantc)
	}
}

func splitIntoPackages(t *testing.T, dir string) (incompatibles, compatibles []string) {
	// Read the input file line by line.
	// Write a line into the old or new package,
	// dependent on comments.
	// Also collect expected messages.
	f, err := os.Open("testdata/tests.go")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	if err := os.MkdirAll(filepath.Join(dir, "src", "apidiff"), 0700); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(filepath.Join(dir, "src", "apidiff", "go.mod"), []byte("module apidiff\n"), 0666); err != nil {
		t.Fatal(err)
	}

	oldd := filepath.Join(dir, "src/apidiff/old")
	newd := filepath.Join(dir, "src/apidiff/new")
	if err := os.MkdirAll(oldd, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(newd, 0700); err != nil && !os.IsExist(err) {
		t.Fatal(err)
	}

	oldf, err := os.Create(filepath.Join(oldd, "old.go"))
	if err != nil {
		t.Fatal(err)
	}
	newf, err := os.Create(filepath.Join(newd, "new.go"))
	if err != nil {
		t.Fatal(err)
	}

	wl := func(f *os.File, line string) {
		if _, err := fmt.Fprintln(f, line); err != nil {
			t.Fatal(err)
		}
	}
	writeBoth := func(line string) { wl(oldf, line); wl(newf, line) }
	writeln := writeBoth
	s := bufio.NewScanner(f)
	for s.Scan() {
		line := s.Text()
		tl := strings.TrimSpace(line)
		switch {
		case tl == "// old":
			writeln = func(line string) { wl(oldf, line) }
		case tl == "// new":
			writeln = func(line string) { wl(newf, line) }
		case tl == "// both":
			writeln = writeBoth
		case strings.HasPrefix(tl, "// i "):
			incompatibles = append(incompatibles, strings.TrimSpace(tl[4:]))
		case strings.HasPrefix(tl, "// c "):
			compatibles = append(compatibles, strings.TrimSpace(tl[4:]))
		default:
			writeln(line)
		}
	}
	if s.Err() != nil {
		t.Fatal(s.Err())
	}
	return
}

func load(importPath, goPath string) (*packages.Package, error) {
	cfg := &packages.Config{
		Mode: packages.LoadTypes,
	}
	if goPath != "" {
		cfg.Env = append(os.Environ(), "GOPATH="+goPath)
		cfg.Dir = filepath.Join(goPath, "src", filepath.FromSlash(importPath))
	}
	pkgs, err := packages.Load(cfg, importPath)
	if err != nil {
		return nil, err
	}
	if len(pkgs[0].Errors) > 0 {
		return nil, pkgs[0].Errors[0]
	}
	return pkgs[0], nil
}

func TestExportedFields(t *testing.T) {
	pkg, err := load("golang.org/x/tools/internal/apidiff/testdata/exported_fields", "")
	if err != nil {
		t.Fatal(err)
	}
	typeof := func(name string) types.Type {
		return pkg.Types.Scope().Lookup(name).Type()
	}

	s := typeof("S")
	su := s.(*types.Named).Underlying().(*types.Struct)

	ef := exportedSelectableFields(su)
	wants := []struct {
		name string
		typ  types.Type
	}{
		{"A1", typeof("A1")},
		{"D", types.Typ[types.Bool]},
		{"E", types.Typ[types.Int]},
		{"F", typeof("F")},
		{"S", types.NewPointer(s)},
	}

	if got, want := len(ef), len(wants); got != want {
		t.Errorf("got %d fields, want %d\n%+v", got, want, ef)
	}
	for _, w := range wants {
		if got := ef[w.name]; got != nil && !types.Identical(got.Type(), w.typ) {
			t.Errorf("%s: got %v, want %v", w.name, got.Type(), w.typ)
		}
	}
}
