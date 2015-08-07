// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package imports

import (
	"fmt"
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"os"
	"path"
	"path/filepath"
	"strings"
	"sync"

	"golang.org/x/tools/go/ast/astutil"
)

// importToGroup is a list of functions which map from an import path to
// a group number.
var importToGroup = []func(importPath string) (num int, ok bool){
	func(importPath string) (num int, ok bool) {
		if strings.HasPrefix(importPath, "appengine") {
			return 2, true
		}
		return
	},
	func(importPath string) (num int, ok bool) {
		if strings.Contains(importPath, ".") {
			return 1, true
		}
		return
	},
}

func importGroup(importPath string) int {
	for _, fn := range importToGroup {
		if n, ok := fn(importPath); ok {
			return n
		}
	}
	return 0
}

func fixImports(fset *token.FileSet, f *ast.File) (added []string, err error) {
	// refs are a set of possible package references currently unsatisfied by imports.
	// first key: either base package (e.g. "fmt") or renamed package
	// second key: referenced package symbol (e.g. "Println")
	refs := make(map[string]map[string]bool)

	// decls are the current package imports. key is base package or renamed package.
	decls := make(map[string]*ast.ImportSpec)

	// collect potential uses of packages.
	var visitor visitFn
	visitor = visitFn(func(node ast.Node) ast.Visitor {
		if node == nil {
			return visitor
		}
		switch v := node.(type) {
		case *ast.ImportSpec:
			if v.Name != nil {
				decls[v.Name.Name] = v
			} else {
				local := importPathToName(strings.Trim(v.Path.Value, `\"`))
				decls[local] = v
			}
		case *ast.SelectorExpr:
			xident, ok := v.X.(*ast.Ident)
			if !ok {
				break
			}
			if xident.Obj != nil {
				// if the parser can resolve it, it's not a package ref
				break
			}
			pkgName := xident.Name
			if refs[pkgName] == nil {
				refs[pkgName] = make(map[string]bool)
			}
			if decls[pkgName] == nil {
				refs[pkgName][v.Sel.Name] = true
			}
		}
		return visitor
	})
	ast.Walk(visitor, f)

	// Nil out any unused ImportSpecs, to be removed in following passes
	unusedImport := map[string]bool{}
	for pkg, is := range decls {
		if refs[pkg] == nil && pkg != "_" && pkg != "." {
			unusedImport[strings.Trim(is.Path.Value, `"`)] = true
		}
	}
	for ipath := range unusedImport {
		if ipath == "C" {
			// Don't remove cgo stuff.
			continue
		}
		astutil.DeleteImport(fset, f, ipath)
	}

	// Search for imports matching potential package references.
	searches := 0
	type result struct {
		ipath string
		name  string
		err   error
	}
	results := make(chan result)
	for pkgName, symbols := range refs {
		if len(symbols) == 0 {
			continue // skip over packages already imported
		}
		go func(pkgName string, symbols map[string]bool) {
			ipath, rename, err := findImport(pkgName, symbols)
			r := result{ipath: ipath, err: err}
			if rename {
				r.name = pkgName
			}
			results <- r
		}(pkgName, symbols)
		searches++
	}
	for i := 0; i < searches; i++ {
		result := <-results
		if result.err != nil {
			return nil, result.err
		}
		if result.ipath != "" {
			if result.name != "" {
				astutil.AddNamedImport(fset, f, result.name, result.ipath)
			} else {
				astutil.AddImport(fset, f, result.ipath)
			}
			added = append(added, result.ipath)
		}
	}

	return added, nil
}

// importPathToName returns the package name for the given import path.
var importPathToName = importPathToNameGoPath

// importPathToNameBasic assumes the package name is the base of import path.
func importPathToNameBasic(importPath string) (packageName string) {
	return path.Base(importPath)
}

// importPathToNameGoPath finds out the actual package name, as declared in its .go files.
// If there's a problem, it falls back to using importPathToNameBasic.
func importPathToNameGoPath(importPath string) (packageName string) {
	if buildPkg, err := build.Import(importPath, "", 0); err == nil {
		return buildPkg.Name
	} else {
		return importPathToNameBasic(importPath)
	}
}

type pkg struct {
	importpath string // full pkg import path, e.g. "net/http"
	dir        string // absolute file path to pkg directory e.g. "/usr/lib/go/src/fmt"
}

var pkgIndexOnce sync.Once

var pkgIndex struct {
	sync.Mutex
	m map[string][]pkg // shortname => []pkg, e.g "http" => "net/http"
}

// gate is a semaphore for limiting concurrency.
type gate chan struct{}

func (g gate) enter() { g <- struct{}{} }
func (g gate) leave() { <-g }

// fsgate protects the OS & filesystem from too much concurrency.
// Too much disk I/O -> too many threads -> swapping and bad scheduling.
var fsgate = make(gate, 8)

func loadPkgIndex() {
	pkgIndex.Lock()
	pkgIndex.m = make(map[string][]pkg)
	pkgIndex.Unlock()

	var wg sync.WaitGroup
	for _, path := range build.Default.SrcDirs() {
		fsgate.enter()
		f, err := os.Open(path)
		if err != nil {
			fsgate.leave()
			fmt.Fprint(os.Stderr, err)
			continue
		}
		children, err := f.Readdir(-1)
		f.Close()
		fsgate.leave()
		if err != nil {
			fmt.Fprint(os.Stderr, err)
			continue
		}
		for _, child := range children {
			if child.IsDir() {
				wg.Add(1)
				go func(path, name string) {
					defer wg.Done()
					loadPkg(&wg, path, name)
				}(path, child.Name())
			}
		}
	}
	wg.Wait()
}

func loadPkg(wg *sync.WaitGroup, root, pkgrelpath string) {
	importpath := filepath.ToSlash(pkgrelpath)
	dir := filepath.Join(root, importpath)

	fsgate.enter()
	defer fsgate.leave()
	pkgDir, err := os.Open(dir)
	if err != nil {
		return
	}
	children, err := pkgDir.Readdir(-1)
	pkgDir.Close()
	if err != nil {
		return
	}
	// hasGo tracks whether a directory actually appears to be a
	// Go source code directory. If $GOPATH == $HOME, and
	// $HOME/src has lots of other large non-Go projects in it,
	// then the calls to importPathToName below can be expensive.
	hasGo := false
	for _, child := range children {
		// Avoid .foo, _foo, and testdata directory trees.
		name := child.Name()
		if name == "" || name[0] == '.' || name[0] == '_' || name == "testdata" {
			continue
		}
		if strings.HasSuffix(name, ".go") {
			hasGo = true
		}
		if child.IsDir() {
			wg.Add(1)
			go func(root, name string) {
				defer wg.Done()
				loadPkg(wg, root, name)
			}(root, filepath.Join(importpath, name))
		}
	}
	if hasGo {
		shortName := importPathToName(importpath)
		pkgIndex.Lock()
		pkgIndex.m[shortName] = append(pkgIndex.m[shortName], pkg{
			importpath: importpath,
			dir:        dir,
		})
		pkgIndex.Unlock()
	}

}

// loadExports returns a list exports for a package.
var loadExports = loadExportsGoPath

func loadExportsGoPath(dir string) map[string]bool {
	exports := make(map[string]bool)
	buildPkg, err := build.ImportDir(dir, 0)
	if err != nil {
		if strings.Contains(err.Error(), "no buildable Go source files in") {
			return nil
		}
		fmt.Fprintf(os.Stderr, "could not import %q: %v\n", dir, err)
		return nil
	}
	fset := token.NewFileSet()
	for _, files := range [...][]string{buildPkg.GoFiles, buildPkg.CgoFiles} {
		for _, file := range files {
			f, err := parser.ParseFile(fset, filepath.Join(dir, file), nil, 0)
			if err != nil {
				fmt.Fprintf(os.Stderr, "could not parse %q: %v\n", file, err)
				continue
			}
			for name := range f.Scope.Objects {
				if ast.IsExported(name) {
					exports[name] = true
				}
			}
		}
	}
	return exports
}

// findImport searches for a package with the given symbols.
// If no package is found, findImport returns "".
// Declared as a variable rather than a function so goimports can be easily
// extended by adding a file with an init function.
var findImport = findImportGoPath

func findImportGoPath(pkgName string, symbols map[string]bool) (string, bool, error) {
	// Fast path for the standard library.
	// In the common case we hopefully never have to scan the GOPATH, which can
	// be slow with moving disks.
	if pkg, rename, ok := findImportStdlib(pkgName, symbols); ok {
		return pkg, rename, nil
	}

	// TODO(sameer): look at the import lines for other Go files in the
	// local directory, since the user is likely to import the same packages
	// in the current Go file.  Return rename=true when the other Go files
	// use a renamed package that's also used in the current file.

	pkgIndexOnce.Do(loadPkgIndex)

	// Collect exports for packages with matching names.
	var wg sync.WaitGroup
	var pkgsMu sync.Mutex // guards pkgs
	// full importpath => exported symbol => True
	// e.g. "net/http" => "Client" => True
	pkgs := make(map[string]map[string]bool)
	pkgIndex.Lock()
	for _, pkg := range pkgIndex.m[pkgName] {
		wg.Add(1)
		go func(importpath, dir string) {
			defer wg.Done()
			exports := loadExports(dir)
			if exports != nil {
				pkgsMu.Lock()
				pkgs[importpath] = exports
				pkgsMu.Unlock()
			}
		}(pkg.importpath, pkg.dir)
	}
	pkgIndex.Unlock()
	wg.Wait()

	// Filter out packages missing required exported symbols.
	for symbol := range symbols {
		for importpath, exports := range pkgs {
			if !exports[symbol] {
				delete(pkgs, importpath)
			}
		}
	}
	if len(pkgs) == 0 {
		return "", false, nil
	}

	// If there are multiple candidate packages, the shortest one wins.
	// This is a heuristic to prefer the standard library (e.g. "bytes")
	// over e.g. "github.com/foo/bar/bytes".
	shortest := ""
	for importPath := range pkgs {
		if shortest == "" || len(importPath) < len(shortest) {
			shortest = importPath
		}
	}
	return shortest, false, nil
}

type visitFn func(node ast.Node) ast.Visitor

func (fn visitFn) Visit(node ast.Node) ast.Visitor {
	return fn(node)
}

func findImportStdlib(shortPkg string, symbols map[string]bool) (importPath string, rename, ok bool) {
	for symbol := range symbols {
		path := stdlib[shortPkg+"."+symbol]
		if path == "" {
			return "", false, false
		}
		if importPath != "" && importPath != path {
			// Ambiguous. Symbols pointed to different things.
			return "", false, false
		}
		importPath = path
	}
	return importPath, false, importPath != ""
}
