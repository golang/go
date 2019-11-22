// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The code in this file is based largely on the code in
// cmd/guru/implements.go. The guru implementation supports
// looking up "implementers" of methods also, but that
// code has been cut out here for now for simplicity.

package source

import (
	"context"
	"fmt"
	"go/types"
	"sort"

	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/internal/lsp/protocol"
)

func (i *IdentifierInfo) Implementation(ctx context.Context) ([]protocol.Location, error) {
	res, err := i.implementations(ctx)
	if err != nil {
		return nil, err
	}

	var objs []types.Object
	pkgs := map[types.Object]Package{}

	if res.toMethod != nil {
		// If we looked up a method, results are in toMethod.
		for _, s := range res.toMethod {
			// Determine package of receiver.
			recv := s.Recv()
			if p, ok := recv.(*types.Pointer); ok {
				recv = p.Elem()
			}
			if n, ok := recv.(*types.Named); ok {
				pkg := res.pkgs[n]
				pkgs[s.Obj()] = pkg
			}
			// Add object to objs.
			objs = append(objs, s.Obj())
		}
	} else {
		// Otherwise, the results are in to.
		for _, t := range res.to {
			// We'll provide implementations that are named types and pointers to named types.
			if p, ok := t.(*types.Pointer); ok {
				t = p.Elem()
			}
			if n, ok := t.(*types.Named); ok {
				pkg := res.pkgs[n]
				objs = append(objs, n.Obj())
				pkgs[n.Obj()] = pkg
			}
		}
	}

	var locations []protocol.Location

	for _, obj := range objs {
		pkg := pkgs[obj]
		if pkgs[obj] == nil || len(pkg.CompiledGoFiles()) == 0 {
			continue
		}
		file, _, _, err := i.Snapshot.View().FindPosInPackage(pkgs[obj], obj.Pos())
		if err != nil {
			return nil, err
		}
		ident, err := findIdentifier(i.Snapshot, pkg, file, obj.Pos())
		if err != nil {
			return nil, err
		}
		decRange, err := ident.Declaration.Range()
		if err != nil {
			return nil, err
		}
		locations = append(locations, protocol.Location{
			URI:   protocol.NewURI(ident.Declaration.URI()),
			Range: decRange,
		})
	}

	return locations, nil
}
func (i *IdentifierInfo) implementations(ctx context.Context) (implementsResult, error) {
	var T types.Type
	var method *types.Func
	if i.Type.Object == nil {
		// This isn't a type. Is it a method?
		obj, ok := i.Declaration.obj.(*types.Func)
		if !ok {
			return implementsResult{}, fmt.Errorf("no type info object for identifier %q", i.Name)
		}
		recv := obj.Type().(*types.Signature).Recv()
		if recv == nil {
			return implementsResult{}, fmt.Errorf("this function is not a method")
		}
		method = obj
		T = recv.Type()
	} else {
		T = i.Type.Object.Type()
	}

	// Find all named types, even local types (which can have
	// methods due to promotion) and the built-in "error".
	// We ignore aliases 'type M = N' to avoid duplicate
	// reporting of the Named type N.
	var allNamed []*types.Named
	pkgs := map[*types.Named]Package{}
	for _, pkg := range i.Snapshot.KnownPackages(ctx) {
		info := pkg.GetTypesInfo()
		for _, obj := range info.Defs {
			if obj, ok := obj.(*types.TypeName); ok && !obj.IsAlias() {
				if named, ok := obj.Type().(*types.Named); ok {
					allNamed = append(allNamed, named)
					pkgs[named] = pkg
				}
			}
		}
	}

	allNamed = append(allNamed, types.Universe.Lookup("error").Type().(*types.Named))

	var msets typeutil.MethodSetCache

	// TODO(matloob): We only use the to and toMethod result for now. Figure out if we want to
	// surface the from and fromPtr results to users.
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

	// Sort types (arbitrarily) to ensure test determinism.
	sort.Sort(typesByString(to))
	sort.Sort(typesByString(from))
	sort.Sort(typesByString(fromPtr))

	var toMethod []*types.Selection // contain nils
	if method != nil {
		for _, t := range to {
			toMethod = append(toMethod,
				types.NewMethodSet(t).Lookup(method.Pkg(), method.Name()))
		}
	}

	return implementsResult{pkgs, to, from, fromPtr, toMethod}, nil
}

// implementsResult contains the results of an implements query.
type implementsResult struct {
	pkgs     map[*types.Named]Package
	to       []types.Type // named or ptr-to-named types assignable to interface T
	from     []types.Type // named interfaces assignable from T
	fromPtr  []types.Type // named interfaces assignable only from *T
	toMethod []*types.Selection
}

type typesByString []types.Type

func (p typesByString) Len() int           { return len(p) }
func (p typesByString) Less(i, j int) bool { return p[i].String() < p[j].String() }
func (p typesByString) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }
