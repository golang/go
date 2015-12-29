// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.5

package oracle

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"io/ioutil"
	"sort"

	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/oracle/serial"
	"golang.org/x/tools/refactor/importgraph"
)

// Referrers reports all identifiers that resolve to the same object
// as the queried identifier, within any package in the analysis scope.
func referrers(q *Query) error {
	lconf := loader.Config{Build: q.Build}
	allowErrors(&lconf)

	if _, err := importQueryPackage(q.Pos, &lconf); err != nil {
		return err
	}

	var id *ast.Ident
	var obj types.Object
	var lprog *loader.Program
	var pass2 bool
	var qpos *queryPos
	for {
		// Load/parse/type-check the program.
		var err error
		lprog, err = lconf.Load()
		if err != nil {
			return err
		}
		q.Fset = lprog.Fset

		qpos, err = parseQueryPos(lprog, q.Pos, false)
		if err != nil {
			return err
		}

		id, _ = qpos.path[0].(*ast.Ident)
		if id == nil {
			return fmt.Errorf("no identifier here")
		}

		obj = qpos.info.ObjectOf(id)
		if obj == nil {
			// Happens for y in "switch y := x.(type)",
			// the package declaration,
			// and unresolved identifiers.
			if _, ok := qpos.path[1].(*ast.File); ok { // package decl?
				pkg := qpos.info.Pkg
				obj = types.NewPkgName(id.Pos(), pkg, pkg.Name(), pkg)
			} else {
				return fmt.Errorf("no object for identifier: %T", qpos.path[1])
			}
		}

		if pass2 {
			break
		}

		// If the identifier is exported, we must load all packages that
		// depend transitively upon the package that defines it.
		// Treat PkgNames as exported, even though they're lowercase.
		if _, isPkg := obj.(*types.PkgName); !(isPkg || obj.Exported()) {
			break // not exported
		}

		// Scan the workspace and build the import graph.
		// Ignore broken packages.
		_, rev, _ := importgraph.Build(q.Build)

		// Re-load the larger program.
		// Create a new file set so that ...
		// External test packages are never imported,
		// so they will never appear in the graph.
		// (We must reset the Config here, not just reset the Fset field.)
		lconf = loader.Config{
			Fset:  token.NewFileSet(),
			Build: q.Build,
		}
		allowErrors(&lconf)
		for path := range rev.Search(obj.Pkg().Path()) {
			lconf.ImportWithTests(path)
		}
		pass2 = true
	}

	// Iterate over all go/types' Uses facts for the entire program.
	var refs []*ast.Ident
	for _, info := range lprog.AllPackages {
		for id2, obj2 := range info.Uses {
			if sameObj(obj, obj2) {
				refs = append(refs, id2)
			}
		}
	}
	sort.Sort(byNamePos{q.Fset, refs})

	q.result = &referrersResult{
		qpos:  qpos,
		query: id,
		obj:   obj,
		refs:  refs,
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
	qpos  *queryPos
	query *ast.Ident   // identifier of query
	obj   types.Object // object it denotes
	refs  []*ast.Ident // set of all other references to it
}

func (r *referrersResult) display(printf printfFunc) {
	printf(r.obj, "%d references to %s", len(r.refs), r.qpos.objectString(r.obj))

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
		posn := r.qpos.fset.Position(ref.Pos())
		fi := fileinfosByName[posn.Filename]
		if fi == nil {
			fi = &fileinfo{data: make(chan interface{})}
			fileinfosByName[posn.Filename] = fi
			fileinfos = append(fileinfos, fi)

			// First request for this file:
			// start asynchronous read.
			go func() {
				sema <- struct{}{} // acquire token
				content, err := ioutil.ReadFile(posn.Filename)
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

// TODO(adonovan): encode extent, not just Pos info, in Serial form.

func (r *referrersResult) toSerial(res *serial.Result, fset *token.FileSet) {
	referrers := &serial.Referrers{
		Pos:  fset.Position(r.query.Pos()).String(),
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
