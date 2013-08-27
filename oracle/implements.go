package oracle

import (
	"code.google.com/p/go.tools/go/types"
	"code.google.com/p/go.tools/ssa"
)

// Implements displays the 'implements" relation among all
// package-level named types in the package containing the query
// position.
//
// TODO(adonovan): more features:
// - should we include pairs of types belonging to
//   different packages in the 'implements' relation?//
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
func implements(o *oracle) (queryResult, error) {
	pkg := o.prog.Package(o.queryPkgInfo.Pkg)
	if pkg == nil {
		return nil, o.errorf(o.queryPath[0], "no SSA package")
	}

	// Compute set of named interface/concrete types at package level.
	var interfaces, concretes []*types.Named
	for _, mem := range pkg.Members {
		if t, ok := mem.(*ssa.Type); ok {
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

	return &implementsResult{facts}, nil
}

type implementsFact struct {
	iface *types.Named
	conc  types.Type // Named or Pointer(Named)
}

type implementsResult struct {
	facts []implementsFact // facts are grouped by interface
}

func (r *implementsResult) display(o *oracle) {
	// TODO(adonovan): sort to ensure test nondeterminism.
	var prevIface *types.Named
	for _, fact := range r.facts {
		if fact.iface != prevIface {
			o.printf(fact.iface.Obj(), "\tInterface %s:", fact.iface)
			prevIface = fact.iface
		}
		o.printf(deref(fact.conc).(*types.Named).Obj(), "\t\t%s", fact.conc)
	}
}
