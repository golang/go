// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"go/types"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/telemetry/log"
	errors "golang.org/x/xerrors"
)

func (i *IdentifierInfo) Implementation(ctx context.Context) ([]protocol.Location, error) {
	ctx = telemetry.Package.With(ctx, i.pkg.ID())

	impls, err := i.implementations(ctx)
	if err != nil {
		return nil, err
	}

	var locations []protocol.Location
	for _, impl := range impls {
		if impl.pkg == nil || len(impl.pkg.CompiledGoFiles()) == 0 {
			continue
		}

		file, _, err := i.Snapshot.View().FindPosInPackage(impl.pkg, impl.obj.Pos())
		if err != nil {
			log.Error(ctx, "Error getting file for object", err)
			continue
		}

		ident, err := findIdentifier(i.Snapshot, impl.pkg, file, impl.obj.Pos())
		if err != nil {
			log.Error(ctx, "Error getting ident for object", err)
			continue
		}

		decRange, err := ident.Declaration.Range()
		if err != nil {
			log.Error(ctx, "Error getting range for object", err)
			continue
		}

		locations = append(locations, protocol.Location{
			URI:   protocol.NewURI(ident.Declaration.URI()),
			Range: decRange,
		})
	}
	return locations, nil
}

var ErrNotAnInterface = errors.New("not an interface or interface method")

func (i *IdentifierInfo) implementations(ctx context.Context) ([]implementation, error) {
	var (
		T      *types.Interface
		method *types.Func
	)

	switch obj := i.Declaration.obj.(type) {
	case *types.Func:
		method = obj
		if recv := obj.Type().(*types.Signature).Recv(); recv != nil {
			T, _ = recv.Type().Underlying().(*types.Interface)
		}
	case *types.TypeName:
		T, _ = obj.Type().Underlying().(*types.Interface)
	}

	if T == nil {
		return nil, ErrNotAnInterface
	}

	if T.NumMethods() == 0 {
		return nil, nil
	}

	// Find all named types, even local types (which can have methods
	// due to promotion).
	var (
		allNamed []*types.Named
		pkgs     = make(map[*types.Package]Package)
	)
	for _, pkg := range i.Snapshot.KnownPackages(ctx) {
		pkgs[pkg.GetTypes()] = pkg

		info := pkg.GetTypesInfo()
		for _, obj := range info.Defs {
			// We ignore aliases 'type M = N' to avoid duplicate reporting
			// of the Named type N.
			if obj, ok := obj.(*types.TypeName); ok && !obj.IsAlias() {
				// We skip interface types since we only want concrete
				// implementations.
				if named, ok := obj.Type().(*types.Named); ok && !isInterface(named) {
					allNamed = append(allNamed, named)
				}
			}
		}
	}

	var (
		impls []implementation
		seen  = make(map[types.Object]bool)
	)

	// Find all the named types that implement our interface.
	for _, U := range allNamed {
		var concrete types.Type = U
		if !types.AssignableTo(concrete, T) {
			// We also accept T if *T implements our interface.
			concrete = types.NewPointer(concrete)
			if !types.AssignableTo(concrete, T) {
				continue
			}
		}

		var obj types.Object = U.Obj()
		if method != nil {
			obj = types.NewMethodSet(concrete).Lookup(method.Pkg(), method.Name()).Obj()
		}

		if obj == method || seen[obj] {
			continue
		}

		seen[obj] = true

		impls = append(impls, implementation{
			obj: obj,
			pkg: pkgs[obj.Pkg()],
		})
	}

	return impls, nil
}

type implementation struct {
	// obj is the implementation, either a *types.TypeName or *types.Func.
	obj types.Object

	// pkg is the Package that contains obj's definition.
	pkg Package
}
