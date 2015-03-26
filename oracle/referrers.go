// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package oracle

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/token"
	"io/ioutil"
	"sort"

	"golang.org/x/tools/go/types"
	"golang.org/x/tools/oracle/serial"
)

// TODO(adonovan): use golang.org/x/tools/refactor/importgraph to choose
// the scope automatically.

// Referrers reports all identifiers that resolve to the same object
// as the queried identifier, within any package in the analysis scope.
//
func referrers(o *Oracle, qpos *QueryPos) (queryResult, error) {
	id, _ := qpos.path[0].(*ast.Ident)
	if id == nil {
		return nil, fmt.Errorf("no identifier here")
	}

	obj := qpos.info.ObjectOf(id)
	if obj == nil {
		// Happens for y in "switch y := x.(type)", but I think that's all.
		return nil, fmt.Errorf("no object for identifier")
	}

	// Iterate over all go/types' Uses facts for the entire program.
	var refs []*ast.Ident
	for _, info := range o.typeInfo {
		for id2, obj2 := range info.Uses {
			if sameObj(obj, obj2) {
				refs = append(refs, id2)
			}
		}
	}
	sort.Sort(byNamePos(refs))

	return &referrersResult{
		qpos:  qpos,
		query: id,
		obj:   obj,
		refs:  refs,
	}, nil
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

type byNamePos []*ast.Ident

func (p byNamePos) Len() int           { return len(p) }
func (p byNamePos) Less(i, j int) bool { return p[i].NamePos < p[j].NamePos }
func (p byNamePos) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

type referrersResult struct {
	qpos  *QueryPos
	query *ast.Ident   // identifier of query
	obj   types.Object // object it denotes
	refs  []*ast.Ident // set of all other references to it
}

func (r *referrersResult) display(printf printfFunc) {
	printf(r.obj, "%d references to %s", len(r.refs), r.obj)

	// Show referring lines, like grep.
	type fileinfo struct {
		refs     []*ast.Ident
		linenums []int       // line number of refs[i]
		data     chan []byte // file contents
	}
	var fileinfos []*fileinfo
	fileinfosByName := make(map[string]*fileinfo)

	// First pass: start the file reads concurrently.
	for _, ref := range r.refs {
		posn := r.qpos.fset.Position(ref.Pos())
		fi := fileinfosByName[posn.Filename]
		if fi == nil {
			fi = &fileinfo{data: make(chan []byte)}
			fileinfosByName[posn.Filename] = fi
			fileinfos = append(fileinfos, fi)

			// First request for this file:
			// start asynchronous read.
			go func() {
				content, err := ioutil.ReadFile(posn.Filename)
				if err != nil {
					content = []byte(fmt.Sprintf("error: %v", err))
				}
				fi.data <- content
			}()
		}
		fi.refs = append(fi.refs, ref)
		fi.linenums = append(fi.linenums, posn.Line)
	}

	// Second pass: print refs in original order.
	// One line may have several refs at different columns.
	for _, fi := range fileinfos {
		content := <-fi.data // wait for I/O completion
		lines := bytes.Split(content, []byte("\n"))
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
