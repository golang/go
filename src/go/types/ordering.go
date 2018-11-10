// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements resolveOrder.

package types

import (
	"go/ast"
	"sort"
)

// resolveOrder computes the order in which package-level objects
// must be type-checked.
//
// Interface types appear first in the list, sorted topologically
// by dependencies on embedded interfaces that are also declared
// in this package, followed by all other objects sorted in source
// order.
//
// TODO(gri) Consider sorting all types by dependencies here, and
// in the process check _and_ report type cycles. This may simplify
// the full type-checking phase.
//
func (check *Checker) resolveOrder() []Object {
	var ifaces, others []Object

	// collect interface types with their dependencies, and all other objects
	for obj := range check.objMap {
		if ityp := check.interfaceFor(obj); ityp != nil {
			ifaces = append(ifaces, obj)
			// determine dependencies on embedded interfaces
			for _, f := range ityp.Methods.List {
				if len(f.Names) == 0 {
					// Embedded interface: The type must be a (possibly
					// qualified) identifier denoting another interface.
					// Imported interfaces are already fully resolved,
					// so we can ignore qualified identifiers.
					if ident, _ := f.Type.(*ast.Ident); ident != nil {
						embedded := check.pkg.scope.Lookup(ident.Name)
						if check.interfaceFor(embedded) != nil {
							check.objMap[obj].addDep(embedded)
						}
					}
				}
			}
		} else {
			others = append(others, obj)
		}
	}

	// final object order
	var order []Object

	// sort interface types topologically by dependencies,
	// and in source order if there are no dependencies
	sort.Sort(inSourceOrder(ifaces))
	visited := make(objSet)
	for _, obj := range ifaces {
		check.appendInPostOrder(&order, obj, visited)
	}

	// sort everything else in source order
	sort.Sort(inSourceOrder(others))

	return append(order, others...)
}

// interfaceFor returns the AST interface denoted by obj, or nil.
func (check *Checker) interfaceFor(obj Object) *ast.InterfaceType {
	tname, _ := obj.(*TypeName)
	if tname == nil {
		return nil // not a type
	}
	d := check.objMap[obj]
	if d == nil {
		check.dump("%s: %s should have been declared", obj.Pos(), obj.Name())
		unreachable()
	}
	if d.typ == nil {
		return nil // invalid AST - ignore (will be handled later)
	}
	ityp, _ := d.typ.(*ast.InterfaceType)
	return ityp
}

func (check *Checker) appendInPostOrder(order *[]Object, obj Object, visited objSet) {
	if visited[obj] {
		// We've already seen this object; either because it's
		// already added to order, or because we have a cycle.
		// In both cases we stop. Cycle errors are reported
		// when type-checking types.
		return
	}
	visited[obj] = true

	d := check.objMap[obj]
	for _, obj := range orderedSetObjects(d.deps) {
		check.appendInPostOrder(order, obj, visited)
	}

	*order = append(*order, obj)
}

func orderedSetObjects(set objSet) []Object {
	list := make([]Object, len(set))
	i := 0
	for obj := range set {
		// we don't care about the map element value
		list[i] = obj
		i++
	}
	sort.Sort(inSourceOrder(list))
	return list
}

// inSourceOrder implements the sort.Sort interface.
type inSourceOrder []Object

func (a inSourceOrder) Len() int           { return len(a) }
func (a inSourceOrder) Less(i, j int) bool { return a[i].order() < a[j].order() }
func (a inSourceOrder) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
