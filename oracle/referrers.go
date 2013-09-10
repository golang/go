// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package oracle

import (
	"go/ast"
	"go/token"
	"sort"

	"code.google.com/p/go.tools/go/types"
	"code.google.com/p/go.tools/oracle/json"
)

// Referrers reports all identifiers that resolve to the same object
// as the queried identifier, within any package in the analysis scope.
//
func referrers(o *oracle) (queryResult, error) {
	id, _ := o.queryPath[0].(*ast.Ident)
	if id == nil {
		return nil, o.errorf(false, "no identifier here")
	}

	obj := o.queryPkgInfo.ObjectOf(id)
	if obj == nil {
		// Happens for y in "switch y := x.(type)", but I think that's all.
		return nil, o.errorf(false, "no object for identifier")
	}

	obj = primaryPkg(obj)

	// Iterate over all go/types' resolver facts for the entire program.
	var refs []token.Pos
	for _, info := range o.typeInfo {
		for id2, obj2 := range info.Objects {
			obj2 = primaryPkg(obj2)
			if obj2 == obj {
				if id2.NamePos == obj.Pos() {
					continue // skip defining ident
				}
				refs = append(refs, id2.NamePos)
			}
		}
	}
	sort.Sort(byPos(refs))

	return &referrersResult{
		query: id.NamePos,
		obj:   obj,
		refs:  refs,
	}, nil
}

// primaryPkg returns obj unchanged unless it is a (secondary) package
// object created by an ImportSpec, in which case the canonical
// (primary) object is returned.
//
// TODO(adonovan): The need for this function argues against the
// wisdom of the primary/secondary distinction.  Discuss with gri.
//
func primaryPkg(obj types.Object) types.Object {
	if pkg, ok := obj.(*types.Package); ok {
		if prim := pkg.Primary(); prim != nil {
			return prim
		}
	}
	return obj
}

type referrersResult struct {
	query token.Pos    // identifer of query
	obj   types.Object // object it denotes
	refs  []token.Pos  // set of all other references to it
}

func (r *referrersResult) display(printf printfFunc) {
	if r.query != r.obj.Pos() {
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

func (r *referrersResult) toJSON(res *json.Result, fset *token.FileSet) {
	referrers := &json.Referrers{
		Pos:  fset.Position(r.query).String(),
		Desc: r.obj.String(),
	}
	if pos := r.obj.Pos(); pos != token.NoPos { // primary package objects have no Pos()
		referrers.ObjPos = fset.Position(pos).String()
	}
	for _, ref := range r.refs {
		referrers.Refs = append(referrers.Refs, fset.Position(ref).String())
	}
	res.Referrers = referrers
}
