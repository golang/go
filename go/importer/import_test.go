// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package importer

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"testing"
	"time"

	"code.google.com/p/go.tools/go/gcimporter"
	"code.google.com/p/go.tools/go/types"
)

var verbose = flag.Bool("importer.v", false, "verbose mode")

var tests = []string{
	`package p`,

	// consts
	`package p; const X = true`,
	`package p; const X, y, Z = true, false, 0 != 0`,
	`package p; const ( A float32 = 1<<iota; B; C; D)`,
	`package p; const X = "foo"`,
	`package p; const X string = "foo"`,
	`package p; const X = 0`,
	`package p; const X = -42`,
	`package p; const X = 3.14159265`,
	`package p; const X = -1e-10`,
	`package p; const X = 1.2 + 2.3i`,
	`package p; const X = -1i`,
	`package p; import "math"; const Pi = math.Pi`,
	`package p; import m "math"; const Pi = m.Pi`,

	// types
	`package p; type T int`,
	`package p; type T [10]int`,
	`package p; type T []int`,
	`package p; type T struct{}`,
	`package p; type T struct{x int}`,
	`package p; type T *int`,
	`package p; type T func()`,
	`package p; type T *T`,
	`package p; type T interface{}`,
	`package p; type T interface{ foo() }`,
	`package p; type T interface{ m() T }`,
	// TODO(gri) disabled for now - import/export works but
	// types.Type.String() used in the test cannot handle cases
	// like this yet
	// `package p; type T interface{ m() interface{T} }`,
	`package p; type T map[string]bool`,
	`package p; type T chan int`,
	`package p; type T <-chan complex64`,
	`package p; type T chan<- map[int]string`,

	// vars
	`package p; var X int`,
	`package p; var X, Y, Z struct{f int "tag"}`,

	// funcs
	`package p; func F()`,
	`package p; func F(x int, y struct{}) bool`,
	`package p; type T int; func (*T) F(x int, y struct{}) T`,

	// selected special cases
	`package p; type T int`,
	`package p; type T uint8`,
	`package p; type T byte`,
	`package p; type T error`,
	`package p; import "net/http"; type T http.Client`,
	`package p; import "net/http"; type ( T1 http.Client; T2 struct { http.Client } )`,
	`package p; import "unsafe"; type ( T1 unsafe.Pointer; T2 unsafe.Pointer )`,
	`package p; import "unsafe"; type T struct { p unsafe.Pointer }`,
}

func TestImportSrc(t *testing.T) {
	for _, src := range tests {
		pkg, err := pkgForSource(src)
		if err != nil {
			t.Errorf("typecheck failed: %s", err)
			continue
		}
		testExportImport(t, pkg, "")
	}
}

func TestImportStdLib(t *testing.T) {
	start := time.Now()

	libs, err := stdLibs()
	if err != nil {
		t.Fatalf("could not compute list of std libraries: %s", err)
	}

	// make sure printed go/types types and gc-imported types
	// can be compared reasonably well
	types.GcCompatibilityMode = true

	var totSize, totGcsize, n int
	for _, lib := range libs {
		// limit run time for short tests
		if testing.Short() && time.Since(start) >= 750*time.Millisecond {
			return
		}

		pkg, err := pkgForPath(lib)
		switch err := err.(type) {
		case nil:
			// ok
		case *build.NoGoError:
			// no Go files - ignore
			continue
		default:
			t.Errorf("typecheck failed: %s", err)
			continue
		}

		size, gcsize := testExportImport(t, pkg, lib)
		if *verbose {
			fmt.Printf("%s\t%d\t%d\t%d%%\n", lib, size, gcsize, int(float64(size)*100/float64(gcsize)))
		}
		totSize += size
		totGcsize += gcsize
		n++
	}

	if *verbose {
		fmt.Printf("\n%d\t%d\t%d%%\n", totSize, totGcsize, int(float64(totSize)*100/float64(totGcsize)))
	}

	types.GcCompatibilityMode = false
}

func testExportImport(t *testing.T, pkg0 *types.Package, path string) (size, gcsize int) {
	data := ExportData(pkg0)
	size = len(data)

	imports := make(map[string]*types.Package)
	pkg1, err := ImportData(imports, data)
	if err != nil {
		t.Errorf("package %s: import failed: %s", pkg0.Name(), err)
		return
	}

	s0 := pkgString(pkg0)
	s1 := pkgString(pkg1)
	if s1 != s0 {
		t.Errorf("package %s: \nimport got:\n%s\nwant:\n%s\n", pkg0.Name(), s1, s0)
	}

	// If we have a standard library, compare also against the gcimported package.
	if path == "" {
		return // not std library
	}

	gcdata, err := gcExportData(path)
	gcsize = len(gcdata)

	imports = make(map[string]*types.Package)
	pkg2, err := gcImportData(imports, gcdata, path)
	if err != nil {
		t.Errorf("package %s: gcimport failed: %s", pkg0.Name(), err)
		return
	}

	s2 := pkgString(pkg2)
	if s2 != s0 {
		t.Errorf("package %s: \ngcimport got:\n%s\nwant:\n%s\n", pkg0.Name(), s2, s0)
	}

	return
}

func pkgForSource(src string) (*types.Package, error) {
	// parse file
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", src, 0)
	if err != nil {
		return nil, err
	}

	// typecheck file
	return types.Check("import-test", fset, []*ast.File{f})
}

func pkgForPath(path string) (*types.Package, error) {
	// collect filenames
	ctxt := build.Default
	pkginfo, err := ctxt.Import(path, "", 0)
	if err != nil {
		return nil, err
	}
	filenames := append(pkginfo.GoFiles, pkginfo.CgoFiles...)

	// parse files
	fset := token.NewFileSet()
	files := make([]*ast.File, len(filenames))
	for i, filename := range filenames {
		var err error
		files[i], err = parser.ParseFile(fset, filepath.Join(pkginfo.Dir, filename), nil, 0)
		if err != nil {
			return nil, err
		}
	}

	// typecheck files
	// (we only care about exports and thus can ignore function bodies)
	conf := types.Config{IgnoreFuncBodies: true, FakeImportC: true}
	return conf.Check(path, fset, files, nil)
}

// pkgString returns a string representation of a package's exported interface.
func pkgString(pkg *types.Package) string {
	var buf bytes.Buffer

	fmt.Fprintf(&buf, "package %s\n", pkg.Name())

	scope := pkg.Scope()
	for _, name := range scope.Names() {
		if exported(name) {
			obj := scope.Lookup(name)
			buf.WriteString(obj.String())

			switch obj := obj.(type) {
			case *types.Const:
				// For now only print constant values if they are not float
				// or complex. This permits comparing go/types results with
				// gc-generated gcimported package interfaces.
				info := obj.Type().Underlying().(*types.Basic).Info()
				if info&types.IsFloat == 0 && info&types.IsComplex == 0 {
					fmt.Fprintf(&buf, " = %s", obj.Val())
				}

			case *types.TypeName:
				// Print associated methods.
				// Basic types (e.g., unsafe.Pointer) have *types.Basic
				// type rather than *types.Named; so we need to check.
				if typ, _ := obj.Type().(*types.Named); typ != nil {
					if n := typ.NumMethods(); n > 0 {
						// Sort methods by name so that we get the
						// same order independent of whether the
						// methods got imported or coming directly
						// for the source.
						// TODO(gri) This should probably be done
						// in go/types.
						list := make([]*types.Func, n)
						for i := 0; i < n; i++ {
							list[i] = typ.Method(i)
						}
						sort.Sort(byName(list))

						buf.WriteString("\nmethods (\n")
						for _, m := range list {
							fmt.Fprintf(&buf, "\t%s\n", m)
						}
						buf.WriteString(")")
					}
				}
			}
			buf.WriteByte('\n')
		}
	}

	return buf.String()
}

var stdLibRoot = filepath.Join(runtime.GOROOT(), "src", "pkg") + string(filepath.Separator)

// The following std libraries are excluded from the stdLibs list.
var excluded = map[string]bool{
	"builtin": true, // contains type declarations with cycles
	"unsafe":  true, // contains fake declarations
}

// stdLibs returns the list if standard library package paths.
func stdLibs() (list []string, err error) {
	err = filepath.Walk(stdLibRoot, func(path string, info os.FileInfo, err error) error {
		if err == nil && info.IsDir() {
			if info.Name() == "testdata" {
				return filepath.SkipDir
			}
			pkgPath := path[len(stdLibRoot):] // remove stdLibRoot
			if len(pkgPath) > 0 && !excluded[pkgPath] {
				list = append(list, pkgPath)
			}
		}
		return nil
	})
	return
}

type byName []*types.Func

func (a byName) Len() int           { return len(a) }
func (a byName) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a byName) Less(i, j int) bool { return a[i].Name() < a[j].Name() }

// gcExportData returns the gc-generated export data for the given path.
// It is based on a trimmed-down version of gcimporter.Import which does
// not do the actual import, does not handle package unsafe, and assumes
// that path is a correct standard library package path (no canonicalization,
// or handling of local import paths).
func gcExportData(path string) ([]byte, error) {
	filename, id := gcimporter.FindPkg(path, "")
	if filename == "" {
		return nil, fmt.Errorf("can't find import: %s", path)
	}
	if id != path {
		panic("path should be canonicalized")
	}

	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	buf := bufio.NewReader(f)
	if err = gcimporter.FindExportData(buf); err != nil {
		return nil, err
	}

	var data []byte
	for {
		line, err := buf.ReadBytes('\n')
		if err != nil {
			return nil, err
		}
		data = append(data, line...)
		// export data ends in "$$\n"
		if len(line) == 3 && line[0] == '$' && line[1] == '$' {
			return data, nil
		}
	}
}

func gcImportData(imports map[string]*types.Package, data []byte, path string) (*types.Package, error) {
	filename := fmt.Sprintf("<filename for %s>", path) // so we have a decent error message if necessary
	return gcimporter.ImportData(imports, filename, path, bufio.NewReader(bytes.NewBuffer(data)))
}
