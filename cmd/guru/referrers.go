// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"go/types"
	"io"
	"log"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"

	"golang.org/x/tools/cmd/guru/serial"
	"golang.org/x/tools/go/buildutil"
	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/imports"
	"golang.org/x/tools/refactor/importgraph"
)

// The referrers function reports all identifiers that resolve to the same object
// as the queried identifier, within any package in the workspace.
func referrers(q *Query) error {
	fset := token.NewFileSet()
	lconf := loader.Config{Fset: fset, Build: q.Build}
	allowErrors(&lconf)

	if _, err := importQueryPackage(q.Pos, &lconf); err != nil {
		return err
	}

	// Load tests of the query package
	// even if the query location is not in the tests.
	for path := range lconf.ImportPkgs {
		lconf.ImportPkgs[path] = true
	}

	// Load/parse/type-check the query package.
	lprog, err := lconf.Load()
	if err != nil {
		return err
	}

	qpos, err := parseQueryPos(lprog, q.Pos, false)
	if err != nil {
		return err
	}

	id, _ := qpos.path[0].(*ast.Ident)
	if id == nil {
		return fmt.Errorf("no identifier here")
	}

	obj := qpos.info.ObjectOf(id)
	if obj == nil {
		// Happens for y in "switch y := x.(type)",
		// the package declaration,
		// and unresolved identifiers.
		if _, ok := qpos.path[1].(*ast.File); ok { // package decl?
			return packageReferrers(q, qpos.info.Pkg.Path())
		}
		return fmt.Errorf("no object for identifier: %T", qpos.path[1])
	}

	// Imported package name?
	if pkgname, ok := obj.(*types.PkgName); ok {
		return packageReferrers(q, pkgname.Imported().Path())
	}

	if obj.Pkg() == nil {
		return fmt.Errorf("references to predeclared %q are everywhere!", obj.Name())
	}

	q.Output(fset, &referrersInitialResult{
		qinfo: qpos.info,
		obj:   obj,
	})

	// For a globally accessible object defined in package P, we
	// must load packages that depend on P.  Specifically, for a
	// package-level object, we need load only direct importers
	// of P, but for a field or method, we must load
	// any package that transitively imports P.

	if global, pkglevel := classify(obj); global {
		if pkglevel {
			return globalReferrersPkgLevel(q, obj, fset)
		}
		// We'll use the object's position to identify it in the larger program.
		objposn := fset.Position(obj.Pos())
		defpkg := obj.Pkg().Path() // defining package
		return globalReferrers(q, qpos.info.Pkg.Path(), defpkg, objposn)
	}

	outputUses(q, fset, usesOf(obj, qpos.info), obj.Pkg())

	return nil // success
}

// classify classifies objects by how far
// we have to look to find references to them.
func classify(obj types.Object) (global, pkglevel bool) {
	if obj.Exported() {
		if obj.Parent() == nil {
			// selectable object (field or method)
			return true, false
		}
		if obj.Parent() == obj.Pkg().Scope() {
			// lexical object (package-level var/const/func/type)
			return true, true
		}
	}
	// object with unexported named or defined in local scope
	return false, false
}

// packageReferrers reports all references to the specified package
// throughout the workspace.
func packageReferrers(q *Query, path string) error {
	// Scan the workspace and build the import graph.
	// Ignore broken packages.
	_, rev, _ := importgraph.Build(q.Build)

	// Find the set of packages that directly import the query package.
	// Only those packages need typechecking of function bodies.
	users := rev[path]

	// Load the larger program.
	fset := token.NewFileSet()
	lconf := loader.Config{
		Fset:  fset,
		Build: q.Build,
		TypeCheckFuncBodies: func(p string) bool {
			return users[strings.TrimSuffix(p, "_test")]
		},
	}
	allowErrors(&lconf)

	// The importgraph doesn't treat external test packages
	// as separate nodes, so we must use ImportWithTests.
	for path := range users {
		lconf.ImportWithTests(path)
	}

	// Subtle!  AfterTypeCheck needs no mutex for qpkg because the
	// topological import order gives us the necessary happens-before edges.
	// TODO(adonovan): what about import cycles?
	var qpkg *types.Package

	// For efficiency, we scan each package for references
	// just after it has been type-checked.  The loader calls
	// AfterTypeCheck (concurrently), providing us with a stream of
	// packages.
	lconf.AfterTypeCheck = func(info *loader.PackageInfo, files []*ast.File) {
		// AfterTypeCheck may be called twice for the same package due to augmentation.

		if info.Pkg.Path() == path && qpkg == nil {
			// Found the package of interest.
			qpkg = info.Pkg
			fakepkgname := types.NewPkgName(token.NoPos, qpkg, qpkg.Name(), qpkg)
			q.Output(fset, &referrersInitialResult{
				qinfo: info,
				obj:   fakepkgname, // bogus
			})
		}

		// Only inspect packages that directly import the
		// declaring package (and thus were type-checked).
		if lconf.TypeCheckFuncBodies(info.Pkg.Path()) {
			// Find PkgNames that refer to qpkg.
			// TODO(adonovan): perhaps more useful would be to show imports
			// of the package instead of qualified identifiers.
			var refs []*ast.Ident
			for id, obj := range info.Uses {
				if obj, ok := obj.(*types.PkgName); ok && obj.Imported() == qpkg {
					refs = append(refs, id)
				}
			}
			outputUses(q, fset, refs, info.Pkg)
		}

		clearInfoFields(info) // save memory
	}

	lconf.Load() // ignore error

	if qpkg == nil {
		log.Fatalf("query package %q not found during reloading", path)
	}

	return nil
}

func usesOf(queryObj types.Object, info *loader.PackageInfo) []*ast.Ident {
	var refs []*ast.Ident
	for id, obj := range info.Uses {
		if sameObj(queryObj, obj) {
			refs = append(refs, id)
		}
	}
	return refs
}

// outputUses outputs a result describing refs, which appear in the package denoted by info.
func outputUses(q *Query, fset *token.FileSet, refs []*ast.Ident, pkg *types.Package) {
	if len(refs) > 0 {
		sort.Sort(byNamePos{fset, refs})
		q.Output(fset, &referrersPackageResult{
			pkg:   pkg,
			build: q.Build,
			fset:  fset,
			refs:  refs,
		})
	}
}

// globalReferrers reports references throughout the entire workspace to the
// object (a field or method) at the specified source position.
// Its defining package is defpkg, and the query package is qpkg.
func globalReferrers(q *Query, qpkg, defpkg string, objposn token.Position) error {
	// Scan the workspace and build the import graph.
	// Ignore broken packages.
	_, rev, _ := importgraph.Build(q.Build)

	// Find the set of packages that depend on defpkg.
	// Only function bodies in those packages need type-checking.
	users := rev.Search(defpkg) // transitive importers

	// Prepare to load the larger program.
	fset := token.NewFileSet()
	lconf := loader.Config{
		Fset:  fset,
		Build: q.Build,
		TypeCheckFuncBodies: func(p string) bool {
			return users[strings.TrimSuffix(p, "_test")]
		},
	}
	allowErrors(&lconf)

	// The importgraph doesn't treat external test packages
	// as separate nodes, so we must use ImportWithTests.
	for path := range users {
		lconf.ImportWithTests(path)
	}

	// The remainder of this function is somewhat tricky because it
	// operates on the concurrent stream of packages observed by the
	// loader's AfterTypeCheck hook.  Most of guru's helper
	// functions assume the entire program has already been loaded,
	// so we can't use them here.
	// TODO(adonovan): smooth things out once the other changes have landed.

	// Results are reported concurrently from within the
	// AfterTypeCheck hook.  The program may provide a useful stream
	// of information even if the user doesn't let the program run
	// to completion.

	var (
		mu   sync.Mutex
		qobj types.Object
	)

	// For efficiency, we scan each package for references
	// just after it has been type-checked.  The loader calls
	// AfterTypeCheck (concurrently), providing us with a stream of
	// packages.
	lconf.AfterTypeCheck = func(info *loader.PackageInfo, files []*ast.File) {
		// AfterTypeCheck may be called twice for the same package due to augmentation.

		// Only inspect packages that depend on the declaring package
		// (and thus were type-checked).
		if lconf.TypeCheckFuncBodies(info.Pkg.Path()) {
			// Record the query object and its package when we see it.
			mu.Lock()
			if qobj == nil && info.Pkg.Path() == defpkg {
				// Find the object by its position (slightly ugly).
				qobj = findObject(fset, &info.Info, objposn)
				if qobj == nil {
					// It really ought to be there;
					// we found it once already.
					log.Fatalf("object at %s not found in package %s",
						objposn, defpkg)
				}
			}
			obj := qobj
			mu.Unlock()

			// Look for references to the query object.
			if obj != nil {
				outputUses(q, fset, usesOf(obj, info), info.Pkg)
			}
		}

		clearInfoFields(info) // save memory
	}

	lconf.Load() // ignore error

	if qobj == nil {
		log.Fatal("query object not found during reloading")
	}

	return nil // success
}

// globalReferrersPkgLevel reports references throughout the entire workspace to the package-level object obj.
// It assumes that the query object itself has already been reported.
func globalReferrersPkgLevel(q *Query, obj types.Object, fset *token.FileSet) error {
	// globalReferrersPkgLevel uses go/ast and friends instead of go/types.
	// This affords a considerable performance benefit.
	// It comes at the cost of some code complexity.
	//
	// Here's a high level summary.
	//
	// The goal is to find references to the query object p.Q.
	// There are several possible scenarios, each handled differently.
	//
	// 1. We are looking in a package other than p, and p is not dot-imported.
	//    This is the simplest case. Q must be referred to as n.Q,
	//    where n is the name under which p is imported.
	//    We look at all imports of p to gather all names under which it is imported.
	//    (In the typical case, it is imported only once, under its default name.)
	//    Then we look at all selector expressions and report any matches.
	//
	// 2. We are looking in a package other than p, and p is dot-imported.
	//    In this case, Q will be referred to just as Q.
	//    Furthermore, go/ast's object resolution will not be able to resolve
	//    Q to any other object, unlike any local (file- or function- or block-scoped) object.
	//    So we look at all matching identifiers and report all unresolvable ones.
	//
	// 3. We are looking in package p.
	//    (Care must be taken to separate p and p_test (an xtest package),
	//    and make sure that they are treated as separate packages.)
	//    In this case, we give go/ast the entire package for object resolution,
	//    instead of going file by file.
	//    We then iterate over all identifiers that resolve to the query object.
	//    (The query object itself has already been reported, so we don't re-report it.)
	//
	// We always skip all files that don't contain the string Q, as they cannot be
	// relevant to finding references to Q.
	//
	// We parse all files leniently. In the presence of parsing errors, results are best-effort.

	// Scan the workspace and build the import graph.
	// Ignore broken packages.
	_, rev, _ := importgraph.Build(q.Build)

	// Find the set of packages that directly import defpkg.
	defpkg := obj.Pkg().Path()
	defpkg = strings.TrimSuffix(defpkg, "_test") // package x_test actually has package name x
	defpkg = imports.VendorlessPath(defpkg)      // remove vendor goop

	users := rev[defpkg]
	if len(users) == 0 {
		users = make(map[string]bool)
	}
	// We also need to check defpkg itself, and its xtests.
	// For the reverse graph packages, we process xtests with the main package.
	// defpkg gets special handling; we must distinguish between in-package vs out-of-package.
	// To make the control flow below simpler, add defpkg and defpkg xtest placeholders.
	// Use "!test" instead of "_test" because "!" is not a valid character in an import path.
	// (More precisely, it is not guaranteed to be a valid character in an import path,
	// so it is unlikely that it will be in use. See https://golang.org/ref/spec#Import_declarations.)
	users[defpkg] = true
	users[defpkg+"!test"] = true

	cwd, err := os.Getwd()
	if err != nil {
		return err
	}

	defname := obj.Pkg().Name()                    // name of defining package, used for imports using import path only
	isxtest := strings.HasSuffix(defname, "_test") // indicates whether the query object is defined in an xtest package

	name := obj.Name()
	namebytes := []byte(name)          // byte slice version of query object name, for early filtering
	objpos := fset.Position(obj.Pos()) // position of query object, used to prevent re-emitting original decl

	sema := make(chan struct{}, 20) // counting semaphore to limit I/O concurrency
	var wg sync.WaitGroup

	for u := range users {
		u := u
		wg.Add(1)
		go func() {
			defer wg.Done()

			uIsXTest := strings.HasSuffix(u, "!test") // indicates whether this package is the special defpkg xtest package
			u = strings.TrimSuffix(u, "!test")

			// Resolve package.
			sema <- struct{}{} // acquire token
			pkg, err := q.Build.Import(u, cwd, build.IgnoreVendor)
			<-sema // release token
			if err != nil {
				return
			}

			// If we're not in the query package,
			// the object is in another package regardless,
			// so we want to process all files.
			// If we are in the query package,
			// we want to only process the files that are
			// part of that query package;
			// that set depends on whether the query package itself is an xtest.
			inQueryPkg := u == defpkg && isxtest == uIsXTest
			var files []string
			if !inQueryPkg || !isxtest {
				files = append(files, pkg.GoFiles...)
				files = append(files, pkg.TestGoFiles...)
				files = append(files, pkg.CgoFiles...) // use raw cgo files, as we're only parsing
			}
			if !inQueryPkg || isxtest {
				files = append(files, pkg.XTestGoFiles...)
			}

			if len(files) == 0 {
				return
			}

			var deffiles map[string]*ast.File
			if inQueryPkg {
				deffiles = make(map[string]*ast.File)
			}

			buf := new(bytes.Buffer) // reusable buffer for reading files

			for _, file := range files {
				if !buildutil.IsAbsPath(q.Build, file) {
					file = buildutil.JoinPath(q.Build, pkg.Dir, file)
				}
				buf.Reset()
				sema <- struct{}{} // acquire token
				src, err := readFile(q.Build, file, buf)
				<-sema // release token
				if err != nil {
					continue
				}

				// Fast path: If the object's name isn't present anywhere in the source, ignore the file.
				if !bytes.Contains(src, namebytes) {
					continue
				}

				if inQueryPkg {
					// If we're in the query package, we defer final processing until we have
					// parsed all of the candidate files in the package.
					// Best effort; allow errors and use what we can from what remains.
					f, _ := parser.ParseFile(fset, file, src, parser.AllErrors)
					if f != nil {
						deffiles[file] = f
					}
					continue
				}

				// We aren't in the query package. Go file by file.

				// Parse out only the imports, to check whether the defining package
				// was imported, and if so, under what names.
				// Best effort; allow errors and use what we can from what remains.
				f, _ := parser.ParseFile(fset, file, src, parser.ImportsOnly|parser.AllErrors)
				if f == nil {
					continue
				}

				// pkgnames is the set of names by which defpkg is imported in this file.
				// (Multiple imports in the same file are legal but vanishingly rare.)
				pkgnames := make([]string, 0, 1)
				var isdotimport bool
				for _, imp := range f.Imports {
					path, err := strconv.Unquote(imp.Path.Value)
					if err != nil || path != defpkg {
						continue
					}
					switch {
					case imp.Name == nil:
						pkgnames = append(pkgnames, defname)
					case imp.Name.Name == ".":
						isdotimport = true
					default:
						pkgnames = append(pkgnames, imp.Name.Name)
					}
				}
				if len(pkgnames) == 0 && !isdotimport {
					// Defining package not imported, bail.
					continue
				}

				// Re-parse the entire file.
				// Parse errors are ok; we'll do the best we can with a partial AST, if we have one.
				f, _ = parser.ParseFile(fset, file, src, parser.AllErrors)
				if f == nil {
					continue
				}

				// Walk the AST looking for references.
				var refs []*ast.Ident
				ast.Inspect(f, func(n ast.Node) bool {
					// Check selector expressions.
					// If the selector matches the target name,
					// and the expression is one of the names
					// that the defining package was imported under,
					// then we have a match.
					if sel, ok := n.(*ast.SelectorExpr); ok && sel.Sel.Name == name {
						if id, ok := sel.X.(*ast.Ident); ok {
							for _, n := range pkgnames {
								if n == id.Name {
									refs = append(refs, sel.Sel)
									// Don't recurse further, to avoid duplicate entries
									// from the dot import check below.
									return false
								}
							}
						}
					}
					// Dot imports are special.
					// Objects imported from the defining package are placed in the package scope.
					// go/ast does not resolve them to an object.
					// At all other scopes (file, local), go/ast can do the resolution.
					// So we're looking for object-free idents with the right name.
					// The only other way to get something with the right name at the package scope
					// is to *be* the defining package. We handle that case separately (inQueryPkg).
					if isdotimport {
						if id, ok := n.(*ast.Ident); ok && id.Obj == nil && id.Name == name {
							refs = append(refs, id)
							return false
						}
					}
					return true
				})

				// Emit any references we found.
				if len(refs) > 0 {
					q.Output(fset, &referrersPackageResult{
						pkg:   types.NewPackage(pkg.ImportPath, pkg.Name),
						build: q.Build,
						fset:  fset,
						refs:  refs,
					})
				}
			}

			// If we're in the query package, we've now collected all the files in the package.
			// (Or at least the ones that might contain references to the object.)
			// Find and emit refs.
			if inQueryPkg {
				// Bundle the files together into a package.
				// This does package-level object resolution.
				qpkg, _ := ast.NewPackage(fset, deffiles, nil, nil)
				// Look up the query object; we know that it is defined in the package scope.
				pkgobj := qpkg.Scope.Objects[name]
				if pkgobj == nil {
					panic("missing defpkg object for " + defpkg + "." + name)
				}
				// Find all references to the query object.
				var refs []*ast.Ident
				ast.Inspect(qpkg, func(n ast.Node) bool {
					if id, ok := n.(*ast.Ident); ok {
						// Check both that this is a reference to the query object
						// and that it is not the query object itself;
						// the query object itself was already emitted.
						if id.Obj == pkgobj && objpos != fset.Position(id.Pos()) {
							refs = append(refs, id)
							return false
						}
					}
					return true
				})
				if len(refs) > 0 {
					q.Output(fset, &referrersPackageResult{
						pkg:   types.NewPackage(pkg.ImportPath, pkg.Name),
						build: q.Build,
						fset:  fset,
						refs:  refs,
					})
				}
				deffiles = nil // allow GC
			}
		}()
	}

	wg.Wait()

	return nil
}

// findObject returns the object defined at the specified position.
func findObject(fset *token.FileSet, info *types.Info, objposn token.Position) types.Object {
	good := func(obj types.Object) bool {
		if obj == nil {
			return false
		}
		posn := fset.Position(obj.Pos())
		return posn.Filename == objposn.Filename && posn.Offset == objposn.Offset
	}
	for _, obj := range info.Defs {
		if good(obj) {
			return obj
		}
	}
	for _, obj := range info.Implicits {
		if good(obj) {
			return obj
		}
	}
	return nil
}

// same reports whether x and y are identical, or both are PkgNames
// that import the same Package.
//
func sameObj(x, y types.Object) bool {
	if x == y {
		return true
	}
	if x, ok := x.(*types.PkgName); ok {
		if y, ok := y.(*types.PkgName); ok {
			return x.Imported() == y.Imported()
		}
	}
	return false
}

func clearInfoFields(info *loader.PackageInfo) {
	// TODO(adonovan): opt: save memory by eliminating unneeded scopes/objects.
	// (Requires go/types change for Go 1.7.)
	//   info.Pkg.Scope().ClearChildren()

	// Discard the file ASTs and their accumulated type
	// information to save memory.
	info.Files = nil
	info.Defs = make(map[*ast.Ident]types.Object)
	info.Uses = make(map[*ast.Ident]types.Object)
	info.Implicits = make(map[ast.Node]types.Object)

	// Also, disable future collection of wholly unneeded
	// type information for the package in case there is
	// more type-checking to do (augmentation).
	info.Types = nil
	info.Scopes = nil
	info.Selections = nil
}

// -------- utils --------

// An deterministic ordering for token.Pos that doesn't
// depend on the order in which packages were loaded.
func lessPos(fset *token.FileSet, x, y token.Pos) bool {
	fx := fset.File(x)
	fy := fset.File(y)
	if fx != fy {
		return fx.Name() < fy.Name()
	}
	return x < y
}

type byNamePos struct {
	fset *token.FileSet
	ids  []*ast.Ident
}

func (p byNamePos) Len() int      { return len(p.ids) }
func (p byNamePos) Swap(i, j int) { p.ids[i], p.ids[j] = p.ids[j], p.ids[i] }
func (p byNamePos) Less(i, j int) bool {
	return lessPos(p.fset, p.ids[i].NamePos, p.ids[j].NamePos)
}

// referrersInitialResult is the initial result of a "referrers" query.
type referrersInitialResult struct {
	qinfo *loader.PackageInfo
	obj   types.Object // object it denotes
}

func (r *referrersInitialResult) PrintPlain(printf printfFunc) {
	printf(r.obj, "references to %s",
		types.ObjectString(r.obj, types.RelativeTo(r.qinfo.Pkg)))
}

func (r *referrersInitialResult) JSON(fset *token.FileSet) []byte {
	var objpos string
	if pos := r.obj.Pos(); pos.IsValid() {
		objpos = fset.Position(pos).String()
	}
	return toJSON(&serial.ReferrersInitial{
		Desc:   r.obj.String(),
		ObjPos: objpos,
	})
}

// referrersPackageResult is the streaming result for one package of a "referrers" query.
type referrersPackageResult struct {
	pkg   *types.Package
	build *build.Context
	fset  *token.FileSet
	refs  []*ast.Ident // set of all other references to it
}

// forEachRef calls f(id, text) for id in r.refs, in order.
// Text is the text of the line on which id appears.
func (r *referrersPackageResult) foreachRef(f func(id *ast.Ident, text string)) {
	// Show referring lines, like grep.
	type fileinfo struct {
		refs     []*ast.Ident
		linenums []int            // line number of refs[i]
		data     chan interface{} // file contents or error
	}
	var fileinfos []*fileinfo
	fileinfosByName := make(map[string]*fileinfo)

	// First pass: start the file reads concurrently.
	sema := make(chan struct{}, 20) // counting semaphore to limit I/O concurrency
	for _, ref := range r.refs {
		posn := r.fset.Position(ref.Pos())
		fi := fileinfosByName[posn.Filename]
		if fi == nil {
			fi = &fileinfo{data: make(chan interface{})}
			fileinfosByName[posn.Filename] = fi
			fileinfos = append(fileinfos, fi)

			// First request for this file:
			// start asynchronous read.
			go func() {
				sema <- struct{}{} // acquire token
				content, err := readFile(r.build, posn.Filename, nil)
				<-sema // release token
				if err != nil {
					fi.data <- err
				} else {
					fi.data <- content
				}
			}()
		}
		fi.refs = append(fi.refs, ref)
		fi.linenums = append(fi.linenums, posn.Line)
	}

	// Second pass: print refs in original order.
	// One line may have several refs at different columns.
	for _, fi := range fileinfos {
		v := <-fi.data // wait for I/O completion

		// Print one item for all refs in a file that could not
		// be loaded (perhaps due to //line directives).
		if err, ok := v.(error); ok {
			var suffix string
			if more := len(fi.refs) - 1; more > 0 {
				suffix = fmt.Sprintf(" (+ %d more refs in this file)", more)
			}
			f(fi.refs[0], err.Error()+suffix)
			continue
		}

		lines := bytes.Split(v.([]byte), []byte("\n"))
		for i, ref := range fi.refs {
			f(ref, string(lines[fi.linenums[i]-1]))
		}
	}
}

// readFile is like ioutil.ReadFile, but
// it goes through the virtualized build.Context.
// If non-nil, buf must have been reset.
func readFile(ctxt *build.Context, filename string, buf *bytes.Buffer) ([]byte, error) {
	rc, err := buildutil.OpenFile(ctxt, filename)
	if err != nil {
		return nil, err
	}
	defer rc.Close()
	if buf == nil {
		buf = new(bytes.Buffer)
	}
	if _, err := io.Copy(buf, rc); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func (r *referrersPackageResult) PrintPlain(printf printfFunc) {
	r.foreachRef(func(id *ast.Ident, text string) {
		printf(id, "%s", text)
	})
}

func (r *referrersPackageResult) JSON(fset *token.FileSet) []byte {
	refs := serial.ReferrersPackage{Package: r.pkg.Path()}
	r.foreachRef(func(id *ast.Ident, text string) {
		refs.Refs = append(refs.Refs, serial.Ref{
			Pos:  fset.Position(id.NamePos).String(),
			Text: text,
		})
	})
	return toJSON(refs)
}
