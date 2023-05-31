// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"reflect"
	"sort"
	"strings"

	"golang.org/x/tools/cmd/guru/serial"
	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/refactor/importgraph"
)

// The implements function displays the "implements" relation as it pertains to the
// selected type.
// If the selection is a method, 'implements' displays
// the corresponding methods of the types that would have been reported
// by an implements query on the receiver type.
func implements(q *Query) error {
	lconf := loader.Config{Build: q.Build}
	allowErrors(&lconf)

	qpkg, err := importQueryPackage(q.Pos, &lconf)
	if err != nil {
		return err
	}

	// Set the packages to search.
	{
		// Otherwise inspect the forward and reverse
		// transitive closure of the selected package.
		// (In theory even this is incomplete.)
		_, rev, _ := importgraph.Build(q.Build)
		for path := range rev.Search(qpkg) {
			lconf.ImportWithTests(path)
		}

		// TODO(adonovan): for completeness, we should also
		// type-check and inspect function bodies in all
		// imported packages.  This would be expensive, but we
		// could optimize by skipping functions that do not
		// contain type declarations.  This would require
		// changing the loader's TypeCheckFuncBodies hook to
		// provide the []*ast.File.
	}

	// Load/parse/type-check the program.
	lprog, err := lconf.Load()
	if err != nil {
		return err
	}

	qpos, err := parseQueryPos(lprog, q.Pos, false)
	if err != nil {
		return err
	}

	// Find the selected type.
	path, action := findInterestingNode(qpos.info, qpos.path)

	var method *types.Func
	var T types.Type // selected type (receiver if method != nil)

	switch action {
	case actionExpr:
		// method?
		if id, ok := path[0].(*ast.Ident); ok {
			if obj, ok := qpos.info.ObjectOf(id).(*types.Func); ok {
				recv := obj.Type().(*types.Signature).Recv()
				if recv == nil {
					return fmt.Errorf("this function is not a method")
				}
				method = obj
				T = recv.Type()
			}
		}

		// If not a method, use the expression's type.
		if T == nil {
			T = qpos.info.TypeOf(path[0].(ast.Expr))
		}

	case actionType:
		T = qpos.info.TypeOf(path[0].(ast.Expr))
	}
	if T == nil {
		return fmt.Errorf("not a type, method, or value")
	}

	// Find all named types, even local types (which can have
	// methods due to promotion) and the built-in "error".
	// We ignore aliases 'type M = N' to avoid duplicate
	// reporting of the Named type N.
	var allNamed []*types.Named
	for _, info := range lprog.AllPackages {
		for _, obj := range info.Defs {
			if obj, ok := obj.(*types.TypeName); ok && !isAlias(obj) {
				if named, ok := obj.Type().(*types.Named); ok {
					allNamed = append(allNamed, named)
				}
			}
		}
	}
	allNamed = append(allNamed, types.Universe.Lookup("error").Type().(*types.Named))

	var msets typeutil.MethodSetCache

	// Test each named type.
	var to, from, fromPtr []types.Type
	for _, U := range allNamed {
		if isInterface(T) {
			if msets.MethodSet(T).Len() == 0 {
				continue // empty interface
			}
			if isInterface(U) {
				if msets.MethodSet(U).Len() == 0 {
					continue // empty interface
				}

				// T interface, U interface
				if !types.Identical(T, U) {
					if types.AssignableTo(U, T) {
						to = append(to, U)
					}
					if types.AssignableTo(T, U) {
						from = append(from, U)
					}
				}
			} else {
				// T interface, U concrete
				if types.AssignableTo(U, T) {
					to = append(to, U)
				} else if pU := types.NewPointer(U); types.AssignableTo(pU, T) {
					to = append(to, pU)
				}
			}
		} else if isInterface(U) {
			if msets.MethodSet(U).Len() == 0 {
				continue // empty interface
			}

			// T concrete, U interface
			if types.AssignableTo(T, U) {
				from = append(from, U)
			} else if pT := types.NewPointer(T); types.AssignableTo(pT, U) {
				fromPtr = append(fromPtr, U)
			}
		}
	}

	var pos interface{} = qpos
	if nt, ok := deref(T).(*types.Named); ok {
		pos = nt.Obj()
	}

	// Sort types (arbitrarily) to ensure test determinism.
	sort.Sort(typesByString(to))
	sort.Sort(typesByString(from))
	sort.Sort(typesByString(fromPtr))

	var toMethod, fromMethod, fromPtrMethod []*types.Selection // contain nils
	if method != nil {
		for _, t := range to {
			toMethod = append(toMethod,
				types.NewMethodSet(t).Lookup(method.Pkg(), method.Name()))
		}
		for _, t := range from {
			fromMethod = append(fromMethod,
				types.NewMethodSet(t).Lookup(method.Pkg(), method.Name()))
		}
		for _, t := range fromPtr {
			fromPtrMethod = append(fromPtrMethod,
				types.NewMethodSet(t).Lookup(method.Pkg(), method.Name()))
		}
	}

	q.Output(lprog.Fset, &implementsResult{
		qpos, T, pos, to, from, fromPtr, method, toMethod, fromMethod, fromPtrMethod,
	})
	return nil
}

type implementsResult struct {
	qpos *queryPos

	t       types.Type   // queried type (not necessarily named)
	pos     interface{}  // pos of t (*types.Name or *QueryPos)
	to      []types.Type // named or ptr-to-named types assignable to interface T
	from    []types.Type // named interfaces assignable from T
	fromPtr []types.Type // named interfaces assignable only from *T

	// if a method was queried:
	method        *types.Func        // queried method
	toMethod      []*types.Selection // method of type to[i], if any
	fromMethod    []*types.Selection // method of type from[i], if any
	fromPtrMethod []*types.Selection // method of type fromPtrMethod[i], if any
}

func (r *implementsResult) PrintPlain(printf printfFunc) {
	relation := "is implemented by"

	meth := func(sel *types.Selection) {
		if sel != nil {
			printf(sel.Obj(), "\t%s method (%s).%s",
				relation, r.qpos.typeString(sel.Recv()), sel.Obj().Name())
		}
	}

	if isInterface(r.t) {
		if types.NewMethodSet(r.t).Len() == 0 { // TODO(adonovan): cache mset
			printf(r.pos, "empty interface type %s", r.qpos.typeString(r.t))
			return
		}

		if r.method == nil {
			printf(r.pos, "interface type %s", r.qpos.typeString(r.t))
		} else {
			printf(r.method, "abstract method %s", r.qpos.objectString(r.method))
		}

		// Show concrete types (or methods) first; use two passes.
		for i, sub := range r.to {
			if !isInterface(sub) {
				if r.method == nil {
					printf(deref(sub).(*types.Named).Obj(), "\t%s %s type %s",
						relation, typeKind(sub), r.qpos.typeString(sub))
				} else {
					meth(r.toMethod[i])
				}
			}
		}
		for i, sub := range r.to {
			if isInterface(sub) {
				if r.method == nil {
					printf(sub.(*types.Named).Obj(), "\t%s %s type %s",
						relation, typeKind(sub), r.qpos.typeString(sub))
				} else {
					meth(r.toMethod[i])
				}
			}
		}

		relation = "implements"
		for i, super := range r.from {
			if r.method == nil {
				printf(super.(*types.Named).Obj(), "\t%s %s",
					relation, r.qpos.typeString(super))
			} else {
				meth(r.fromMethod[i])
			}
		}
	} else {
		relation = "implements"

		if r.from != nil {
			if r.method == nil {
				printf(r.pos, "%s type %s",
					typeKind(r.t), r.qpos.typeString(r.t))
			} else {
				printf(r.method, "concrete method %s",
					r.qpos.objectString(r.method))
			}
			for i, super := range r.from {
				if r.method == nil {
					printf(super.(*types.Named).Obj(), "\t%s %s",
						relation, r.qpos.typeString(super))
				} else {
					meth(r.fromMethod[i])
				}
			}
		}
		if r.fromPtr != nil {
			if r.method == nil {
				printf(r.pos, "pointer type *%s", r.qpos.typeString(r.t))
			} else {
				// TODO(adonovan): de-dup (C).f and (*C).f implementing (I).f.
				printf(r.method, "concrete method %s",
					r.qpos.objectString(r.method))
			}

			for i, psuper := range r.fromPtr {
				if r.method == nil {
					printf(psuper.(*types.Named).Obj(), "\t%s %s",
						relation, r.qpos.typeString(psuper))
				} else {
					meth(r.fromPtrMethod[i])
				}
			}
		} else if r.from == nil {
			printf(r.pos, "%s type %s implements only interface{}",
				typeKind(r.t), r.qpos.typeString(r.t))
		}
	}
}

func (r *implementsResult) JSON(fset *token.FileSet) []byte {
	var method *serial.DescribeMethod
	if r.method != nil {
		method = &serial.DescribeMethod{
			Name: r.qpos.objectString(r.method),
			Pos:  fset.Position(r.method.Pos()).String(),
		}
	}
	return toJSON(&serial.Implements{
		T:                       makeImplementsType(r.t, fset),
		AssignableTo:            makeImplementsTypes(r.to, fset),
		AssignableFrom:          makeImplementsTypes(r.from, fset),
		AssignableFromPtr:       makeImplementsTypes(r.fromPtr, fset),
		AssignableToMethod:      methodsToSerial(r.qpos.info.Pkg, r.toMethod, fset),
		AssignableFromMethod:    methodsToSerial(r.qpos.info.Pkg, r.fromMethod, fset),
		AssignableFromPtrMethod: methodsToSerial(r.qpos.info.Pkg, r.fromPtrMethod, fset),
		Method:                  method,
	})

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

func isInterface(T types.Type) bool { return types.IsInterface(T) }

type typesByString []types.Type

func (p typesByString) Len() int           { return len(p) }
func (p typesByString) Less(i, j int) bool { return p[i].String() < p[j].String() }
func (p typesByString) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }
