// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/build"
	"go/token"
	"go/types"
	"io"
	"log"
	"sort"
	"strings"
	"sync"

	"golang.org/x/tools/cmd/guru/serial"
	"golang.org/x/tools/go/buildutil"
	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/refactor/importgraph"
)

// Referrers reports all identifiers that resolve to the same object
// as the queried identifier, within any package in the workspace.
func referrers(q *Query) error {
	fset := token.NewFileSet()
	lconf := loader.Config{Fset: fset, Build: q.Build}
	allowErrors(&lconf)

	if _, err := importQueryPackage(q.Pos, &lconf); err != nil {
		return err
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

	// For a globally accessible object defined in package P, we
	// must load packages that depend on P.  Specifically, for a
	// package-level object, we need load only direct importers
	// of P, but for a field or interface method, we must load
	// any package that transitively imports P.
	if global, pkglevel := classify(obj); global {
		// We'll use the the object's position to identify it in the larger program.
		objposn := fset.Position(obj.Pos())
		defpkg := obj.Pkg().Path() // defining package
		return globalReferrers(q, qpos.info.Pkg.Path(), defpkg, objposn, pkglevel)
	}

	// Find uses of obj within the query package.
	refs := usesOf(obj, qpos.info)
	sort.Sort(byNamePos{fset, refs})
	q.Fset = fset
	q.result = &referrersResult{
		build: q.Build,
		fset:  fset,
		qinfo: qpos.info,
		obj:   obj,
		refs:  refs,
	}
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

// packageReferrers finds all references to the specified package
// throughout the workspace and populates q.result.
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
	for path := range users {
		lconf.ImportWithTests(path)
	}
	lprog, err := lconf.Load()
	if err != nil {
		return err
	}

	// Find uses of [a fake PkgName that imports] the package.
	//
	// TODO(adonovan): perhaps more useful would be to show imports
	// of the package instead of qualified identifiers.
	qinfo := lprog.Package(path)
	obj := types.NewPkgName(token.NoPos, qinfo.Pkg, qinfo.Pkg.Name(), qinfo.Pkg)
	refs := usesOf(obj, lprog.InitialPackages()...)
	sort.Sort(byNamePos{fset, refs})
	q.Fset = fset
	q.result = &referrersResult{
		build: q.Build,
		fset:  fset,
		qinfo: qinfo,
		obj:   obj,
		refs:  refs,
	}
	return nil
}

// globalReferrers finds references throughout the entire workspace to the
// object at the specified source position.  Its defining package is defpkg,
// and the query package is qpkg.  isPkgLevel indicates whether the object
// is defined at package-level.
func globalReferrers(q *Query, qpkg, defpkg string, objposn token.Position, isPkgLevel bool) error {
	// Scan the workspace and build the import graph.
	// Ignore broken packages.
	_, rev, _ := importgraph.Build(q.Build)

	// Find the set of packages that depend on defpkg.
	// Only function bodies in those packages need type-checking.
	var users map[string]bool
	if isPkgLevel {
		users = rev[defpkg] // direct importers
		if users == nil {
			users = make(map[string]bool)
		}
		users[defpkg] = true // plus the defining package itself
	} else {
		users = rev.Search(defpkg) // transitive importers
	}

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

	var (
		mu    sync.Mutex
		qobj  types.Object
		qinfo *loader.PackageInfo // info for qpkg
	)

	// For efficiency, we scan each package for references
	// just after it has been type-checked.  The loader calls
	// AfterTypeCheck (concurrently), providing us with a stream of
	// packages.
	ch := make(chan []*ast.Ident)
	lconf.AfterTypeCheck = func(info *loader.PackageInfo, files []*ast.File) {
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
				qinfo = info
			}
			obj := qobj
			mu.Unlock()

			// Look for references to the query object.
			if obj != nil {
				ch <- usesOf(obj, info)
			}
		}

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

	go func() {
		lconf.Load() // ignore error
		close(ch)
	}()

	var refs []*ast.Ident
	for ids := range ch {
		refs = append(refs, ids...)
	}
	sort.Sort(byNamePos{fset, refs})

	if qobj == nil {
		log.Fatal("query object not found during reloading")
	}

	// TODO(adonovan): in a follow-up, do away with the
	// analyze/display split so we can print a stream of output
	// directly from the AfterTypeCheck hook.
	// (We should not assume that users let the program run long
	// enough for Load to return.)

	q.Fset = fset
	q.result = &referrersResult{
		build: q.Build,
		fset:  fset,
		qinfo: qinfo,
		obj:   qobj,
		refs:  refs,
	}

	return nil // success
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

// usesOf returns all identifiers in the packages denoted by infos
// that refer to queryObj.
func usesOf(queryObj types.Object, infos ...*loader.PackageInfo) []*ast.Ident {
	var refs []*ast.Ident
	for _, info := range infos {
		for id, obj := range info.Uses {
			if sameObj(queryObj, obj) {
				refs = append(refs, id)
			}
		}
	}
	return refs
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

type referrersResult struct {
	build *build.Context
	fset  *token.FileSet
	qinfo *loader.PackageInfo
	qpos  *queryPos
	obj   types.Object // object it denotes
	refs  []*ast.Ident // set of all other references to it
}

func (r *referrersResult) display(printf printfFunc) {
	printf(r.obj, "%d references to %s",
		len(r.refs), types.ObjectString(r.obj, types.RelativeTo(r.qinfo.Pkg)))

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
				content, err := readFile(r.build, posn.Filename)
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
			printf(fi.refs[0], "%v%s", err, suffix)
			continue
		}

		lines := bytes.Split(v.([]byte), []byte("\n"))
		for i, ref := range fi.refs {
			printf(ref, "%s", lines[fi.linenums[i]-1])
		}
	}
}

// readFile is like ioutil.ReadFile, but
// it goes through the virtualized build.Context.
func readFile(ctxt *build.Context, filename string) ([]byte, error) {
	rc, err := buildutil.OpenFile(ctxt, filename)
	if err != nil {
		return nil, err
	}
	defer rc.Close()
	var buf bytes.Buffer
	if _, err := io.Copy(&buf, rc); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func (r *referrersResult) toSerial(res *serial.Result, fset *token.FileSet) {
	referrers := &serial.Referrers{
		Desc: r.obj.String(),
	}
	if pos := r.obj.Pos(); pos != token.NoPos { // Package objects have no Pos()
		referrers.ObjPos = fset.Position(pos).String()
	}
	for _, ref := range r.refs {
		referrers.Refs = append(referrers.Refs, fset.Position(ref.NamePos).String())
	}
	res.Referrers = referrers
}
