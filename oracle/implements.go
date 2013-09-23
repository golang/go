// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package oracle

import (
	"go/token"

	"code.google.com/p/go.tools/go/types"
	"code.google.com/p/go.tools/oracle/json"
)

// Implements displays the 'implements" relation among all
// package-level named types in the package containing the query
// position.
//
// TODO(adonovan): more features:
// - should we include pairs of types belonging to
//   different packages in the 'implements' relation?
// - should we restrict the query to the type declaration identified
//   by the query position, if any, and use all types in the package
//   otherwise?
// - should we show types that are local to functions?
//   They can only have methods via promotion.
// - abbreviate the set of concrete types implementing the empty
//   interface.
// - should we scan the instruction stream for MakeInterface
//   instructions and report which concrete->interface conversions
//   actually occur, with examples?  (NB: this is not a conservative
//   answer due to ChangeInterface, i.e. subtyping among interfaces.)
//
func implements(o *Oracle, qpos *QueryPos) (queryResult, error) {
	pkg := qpos.info.Pkg

	// Compute set of named interface/concrete types at package level.
	var interfaces, concretes []*types.Named
	scope := pkg.Scope()
	for _, name := range scope.Names() {
		mem := scope.Lookup(name)
		if t, ok := mem.(*types.TypeName); ok {
			nt := t.Type().(*types.Named)
			if _, ok := nt.Underlying().(*types.Interface); ok {
				interfaces = append(interfaces, nt)
			} else {
				concretes = append(concretes, nt)
			}
		}
	}

	// For each interface, show the concrete types that implement it.
	var facts []implementsFact
	for _, iface := range interfaces {
		fact := implementsFact{iface: iface}
		for _, conc := range concretes {
			if types.IsAssignableTo(conc, iface) {
				fact.conc = conc
			} else if ptr := types.NewPointer(conc); types.IsAssignableTo(ptr, iface) {
				fact.conc = ptr
			} else {
				continue
			}
			facts = append(facts, fact)
		}
	}
	// TODO(adonovan): sort facts to ensure test nondeterminism.

	return &implementsResult{o.prog.Fset, facts}, nil
}

type implementsFact struct {
	iface *types.Named
	conc  types.Type // Named or Pointer(Named)
}

type implementsResult struct {
	fset  *token.FileSet
	facts []implementsFact // facts are grouped by interface
}

func (r *implementsResult) display(printf printfFunc) {
	var prevIface *types.Named
	for _, fact := range r.facts {
		if fact.iface != prevIface {
			printf(fact.iface.Obj(), "\tInterface %s:", fact.iface)
			prevIface = fact.iface
		}
		printf(deref(fact.conc).(*types.Named).Obj(), "\t\t%s", fact.conc)
	}
}

func (r *implementsResult) toJSON(res *json.Result, fset *token.FileSet) {
	var facts []*json.Implements
	for _, fact := range r.facts {
		facts = append(facts, &json.Implements{
			I:    fact.iface.String(),
			IPos: fset.Position(fact.iface.Obj().Pos()).String(),
			C:    fact.conc.String(),
			CPos: fset.Position(deref(fact.conc).(*types.Named).Obj().Pos()).String(),
		})
	}
	res.Implements = facts
}
