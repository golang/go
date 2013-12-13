// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package oracle

import (
	"fmt"
	"go/ast"
	"go/token"
	"reflect"
	"sort"
	"strings"

	"code.google.com/p/go.tools/go/types"
	"code.google.com/p/go.tools/oracle/serial"
)

// Implements displays the "implements" relation as it pertains to the
// selected type.
//
func implements(o *Oracle, qpos *QueryPos) (queryResult, error) {
	// Find the selected type.
	// TODO(adonovan): fix: make it work on qualified Idents too.
	path, action := findInterestingNode(qpos.info, qpos.path)
	if action != actionType {
		return nil, fmt.Errorf("no type here")
	}
	T := qpos.info.TypeOf(path[0].(ast.Expr))
	if T == nil {
		return nil, fmt.Errorf("no type here")
	}

	// Find all named types, even local types (which can have
	// methods via promotion) and the built-in "error".
	//
	// TODO(adonovan): include all packages in PTA scope too?
	// i.e. don't reduceScope?
	//
	var allNamed []types.Type
	for _, info := range o.typeInfo {
		for id, obj := range info.Objects {
			if obj, ok := obj.(*types.TypeName); ok && obj.Pos() == id.Pos() {
				allNamed = append(allNamed, obj.Type())
			}
		}
	}
	allNamed = append(allNamed, types.Universe.Lookup("error").Type())

	// Test each named type.
	var to, from, fromPtr []types.Type
	for _, U := range allNamed {
		if isInterface(T) {
			if T.MethodSet().Len() == 0 {
				continue // empty interface
			}
			if isInterface(U) {
				if U.MethodSet().Len() == 0 {
					continue // empty interface
				}

				// T interface, U interface
				if !types.IsIdentical(T, U) {
					if types.IsAssignableTo(U, T) {
						to = append(to, U)
					}
					if types.IsAssignableTo(T, U) {
						from = append(from, U)
					}
				}
			} else {
				// T interface, U concrete
				if types.IsAssignableTo(U, T) {
					to = append(to, U)
				} else if pU := types.NewPointer(U); types.IsAssignableTo(pU, T) {
					to = append(to, pU)
				}
			}
		} else if isInterface(U) {
			if U.MethodSet().Len() == 0 {
				continue // empty interface
			}

			// T concrete, U interface
			if types.IsAssignableTo(T, U) {
				from = append(from, U)
			} else if pT := types.NewPointer(T); types.IsAssignableTo(pT, U) {
				fromPtr = append(fromPtr, U)
			}
		}
	}

	var pos interface{} = qpos
	if nt, ok := deref(T).(*types.Named); ok {
		pos = nt.Obj()
	}

	// Sort types (arbitrarily) to ensure test nondeterminism.
	sort.Sort(typesByString(to))
	sort.Sort(typesByString(from))
	sort.Sort(typesByString(fromPtr))

	return &implementsResult{T, pos, to, from, fromPtr}, nil
}

type implementsResult struct {
	t       types.Type   // queried type (not necessarily named)
	pos     interface{}  // pos of t (*types.Name or *QueryPos)
	to      []types.Type // named or ptr-to-named types assignable to interface T
	from    []types.Type // named interfaces assignable from T
	fromPtr []types.Type // named interfaces assignable only from *T
}

func (r *implementsResult) display(printf printfFunc) {
	if isInterface(r.t) {
		if r.t.MethodSet().Len() == 0 {
			printf(r.pos, "empty interface type %s", r.t)
			return
		}

		printf(r.pos, "interface type %s", r.t)
		// Show concrete types first; use two passes.
		for _, sub := range r.to {
			if !isInterface(sub) {
				printf(deref(sub).(*types.Named).Obj(), "\tis implemented by %s type %s",
					typeKind(sub), sub)
			}
		}
		for _, sub := range r.to {
			if isInterface(sub) {
				printf(deref(sub).(*types.Named).Obj(), "\tis implemented by %s type %s", typeKind(sub), sub)
			}
		}

		for _, super := range r.from {
			printf(super.(*types.Named).Obj(), "\timplements %s", super)
		}
	} else {
		if r.from != nil {
			printf(r.pos, "%s type %s", typeKind(r.t), r.t)
			for _, super := range r.from {
				printf(super.(*types.Named).Obj(), "\timplements %s", super)
			}
		}
		if r.fromPtr != nil {
			printf(r.pos, "pointer type *%s", r.t)
			for _, psuper := range r.fromPtr {
				printf(psuper.(*types.Named).Obj(), "\timplements %s", psuper)
			}
		} else if r.from == nil {
			printf(r.pos, "%s type %s implements only interface{}", typeKind(r.t), r.t)
		}
	}
}

func (r *implementsResult) toSerial(res *serial.Result, fset *token.FileSet) {
	res.Implements = &serial.Implements{
		T:                 makeImplementsType(r.t, fset),
		AssignableTo:      makeImplementsTypes(r.to, fset),
		AssignableFrom:    makeImplementsTypes(r.from, fset),
		AssignableFromPtr: makeImplementsTypes(r.fromPtr, fset),
	}
}

func makeImplementsTypes(tt []types.Type, fset *token.FileSet) []serial.ImplementsType {
	var r []serial.ImplementsType
	for _, t := range tt {
		r = append(r, makeImplementsType(t, fset))
	}
	return r
}

func makeImplementsType(T types.Type, fset *token.FileSet) serial.ImplementsType {
	var pos token.Pos
	if nt, ok := deref(T).(*types.Named); ok { // implementsResult.t may be non-named
		pos = nt.Obj().Pos()
	}
	return serial.ImplementsType{
		Name: T.String(),
		Pos:  fset.Position(pos).String(),
		Kind: typeKind(T),
	}
}

// typeKind returns a string describing the underlying kind of type,
// e.g. "slice", "array", "struct".
func typeKind(T types.Type) string {
	s := reflect.TypeOf(T.Underlying()).String()
	return strings.ToLower(strings.TrimPrefix(s, "*types."))
}

func isInterface(T types.Type) bool {
	_, isI := T.Underlying().(*types.Interface)
	return isI
}

type typesByString []types.Type

func (p typesByString) Len() int           { return len(p) }
func (p typesByString) Less(i, j int) bool { return p[i].String() < p[j].String() }
func (p typesByString) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }
