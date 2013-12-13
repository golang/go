// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package oracle

import (
	"fmt"
	"go/ast"
	"go/token"
	"sort"

	"code.google.com/p/go.tools/go/types"
	"code.google.com/p/go.tools/oracle/serial"
)

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

	// Iterate over all go/types' resolver facts for the entire program.
	var refs []*ast.Ident
	for _, info := range o.typeInfo {
		for id2, obj2 := range info.Objects {
			if sameObj(obj, obj2) {
				if id2.NamePos == obj.Pos() {
					continue // skip defining ident
				}
				refs = append(refs, id2)
			}
		}
	}
	sort.Sort(byNamePos(refs))

	return &referrersResult{
		query: id,
		obj:   obj,
		refs:  refs,
	}, nil
}

// same reports whether x and y are identical, or both are PkgNames
// referring to the same Package.
//
func sameObj(x, y types.Object) bool {
	if x == y {
		return true
	}
	if _, ok := x.(*types.PkgName); ok {
		if _, ok := y.(*types.PkgName); ok {
			return x.Pkg() == y.Pkg()
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
	query *ast.Ident   // identifer of query
	obj   types.Object // object it denotes
	refs  []*ast.Ident // set of all other references to it
}

func (r *referrersResult) display(printf printfFunc) {
	if r.query.Pos() != r.obj.Pos() {
		printf(r.query, "reference to %s", r.obj.Name())
	}
	// TODO(adonovan): pretty-print object using same logic as
	// (*describeValueResult).display.
	printf(r.obj, "defined here as %s", r.obj)
	for _, ref := range r.refs {
		if r.query != ref {
			printf(ref, "referenced here")
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
