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
	"go/token"
	"go/types"

	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/telemetry/log"
	errors "golang.org/x/xerrors"
)

func (i *IdentifierInfo) Implementation(ctx context.Context) ([]protocol.Location, error) {
	ctx = telemetry.Package.With(ctx, i.pkg.ID())

	res, err := i.implementations(ctx)
	if err != nil {
		return nil, err
	}

	var objs []types.Object
	pkgs := map[token.Pos]Package{}

	if res.toMethod != nil {
		// If we looked up a method, results are in toMethod.
		for _, s := range res.toMethod {
			if pkgs[s.Obj().Pos()] != nil {
				continue
			}
			// Determine package of receiver.
			recv := s.Recv()
			if p, ok := recv.(*types.Pointer); ok {
				recv = p.Elem()
			}
			if n, ok := recv.(*types.Named); ok {
				pkg := res.pkgs[n]
				pkgs[s.Obj().Pos()] = pkg
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
				if pkgs[n.Obj().Pos()] != nil {
					continue
				}
				pkg := res.pkgs[n]
				pkgs[n.Obj().Pos()] = pkg
				objs = append(objs, n.Obj())
			}
		}
	}

	var locations []protocol.Location
	for _, obj := range objs {
		pkg := pkgs[obj.Pos()]
		if pkgs[obj.Pos()] == nil || len(pkg.CompiledGoFiles()) == 0 {
			continue
		}
		file, _, err := i.Snapshot.View().FindPosInPackage(pkgs[obj.Pos()], obj.Pos())
		if err != nil {
			log.Error(ctx, "Error getting file for object", err)
			continue
		}
		ident, err := findIdentifier(i.Snapshot, pkg, file, obj.Pos())
		if err != nil {
			log.Error(ctx, "Error getting ident for object", err)
			continue
		}
		decRange, err := ident.Declaration.Range()
		if err != nil {
			log.Error(ctx, "Error getting range for object", err)
			continue
		}
		// Do not add interface itself to the list.
		if ident.Declaration.spanRange == i.Declaration.spanRange {
			continue
		}
		locations = append(locations, protocol.Location{
			URI:   protocol.NewURI(ident.Declaration.URI()),
			Range: decRange,
		})
	}
	return locations, nil
}

var ErrNotAMethod = errors.New("this function is not a method")

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
			return implementsResult{}, ErrNotAMethod
		}
		method = obj
		T = recv.Type()
	} else {
		T = i.Type.Object.Type()
	}

	// Find all named types, even local types (which can have methods
	// due to promotion). We ignore aliases 'type M = N' to avoid
	// duplicate reporting of the Named type N.
	var allNamed []*types.Named
	pkgs := map[*types.Named]Package{}
	for _, pkg := range i.Snapshot.KnownPackages(ctx) {
		info := pkg.GetTypesInfo()
		for _, obj := range info.Defs {
			if obj, ok := obj.(*types.TypeName); ok && !obj.IsAlias() {
				if named, ok := obj.Type().(*types.Named); ok && !isInterface(named) {
					allNamed = append(allNamed, named)
					pkgs[named] = pkg
				}
			}
		}
	}

	var msets typeutil.MethodSetCache

	// Test each named type.
	var to []types.Type
	for _, U := range allNamed {
		if isInterface(T) {
			if msets.MethodSet(T).Len() == 0 {
				continue // empty interface
			}

			// T interface, U concrete
			if types.AssignableTo(U, T) {
				to = append(to, U)
			} else if pU := types.NewPointer(U); types.AssignableTo(pU, T) {
				to = append(to, pU)
			}
		}
	}
	var toMethod []*types.Selection // contain nils
	if method != nil {
		for _, t := range to {
			toMethod = append(toMethod,
				types.NewMethodSet(t).Lookup(method.Pkg(), method.Name()))
		}
	}
	return implementsResult{pkgs, to, toMethod}, nil
}

// implementsResult contains the results of an implements query.
type implementsResult struct {
	pkgs     map[*types.Named]Package
	to       []types.Type // named or ptr-to-named types assignable to interface T
	toMethod []*types.Selection
}
