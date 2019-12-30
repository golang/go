// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/telemetry/log"
	"golang.org/x/tools/internal/telemetry/trace"
	errors "golang.org/x/xerrors"
)

func Implementation(ctx context.Context, s Snapshot, f FileHandle, pp protocol.Position) ([]protocol.Location, error) {
	ctx, done := trace.StartSpan(ctx, "source.Implementation")
	defer done()

	impls, err := implementations(ctx, s, f, pp)
	if err != nil {
		return nil, err
	}

	var locations []protocol.Location
	for _, impl := range impls {
		if impl.pkg == nil || len(impl.pkg.CompiledGoFiles()) == 0 {
			continue
		}

		rng, err := objToMappedRange(s.View(), impl.pkg, impl.obj)
		if err != nil {
			log.Error(ctx, "Error getting range for object", err)
			continue
		}

		pr, err := rng.Range()
		if err != nil {
			log.Error(ctx, "Error getting protocol range for object", err)
			continue
		}

		locations = append(locations, protocol.Location{
			URI:   protocol.NewURI(rng.URI()),
			Range: pr,
		})
	}
	return locations, nil
}

var ErrNotAnInterface = errors.New("not an interface or interface method")

func implementations(ctx context.Context, s Snapshot, f FileHandle, pp protocol.Position) ([]implementation, error) {

	var (
		impls []implementation
		seen  = make(map[token.Position]bool)
		fset  = s.View().Session().Cache().FileSet()
	)

	objs, err := objectsAtProtocolPos(ctx, s, f, pp)
	if err != nil {
		return nil, err
	}

	for _, obj := range objs {
		var (
			T      *types.Interface
			method *types.Func
		)

		switch obj := obj.(type) {
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
		for _, ph := range s.KnownPackages(ctx) {
			pkg, err := ph.Check(ctx)
			if err != nil {
				return nil, err
			}
			pkgs[pkg.GetTypes()] = pkg

			info := pkg.GetTypesInfo()
			for _, obj := range info.Defs {
				obj, ok := obj.(*types.TypeName)
				// We ignore aliases 'type M = N' to avoid duplicate reporting
				// of the Named type N.
				if !ok || obj.IsAlias() {
					continue
				}
				named, ok := obj.Type().(*types.Named)
				// We skip interface types since we only want concrete
				// implementations.
				if !ok || isInterface(named) {
					continue
				}
				allNamed = append(allNamed, named)
			}
		}

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

			pos := fset.Position(obj.Pos())
			if obj == method || seen[pos] {
				continue
			}

			seen[pos] = true

			impls = append(impls, implementation{
				obj: obj,
				pkg: pkgs[obj.Pkg()],
			})
		}
	}

	return impls, nil
}

type implementation struct {
	// obj is the implementation, either a *types.TypeName or *types.Func.
	obj types.Object

	// pkg is the Package that contains obj's definition.
	pkg Package
}

// objectsAtProtocolPos returns all the type.Objects referenced at the given position.
// An object will be returned for every package that the file belongs to.
func objectsAtProtocolPos(ctx context.Context, s Snapshot, f FileHandle, pp protocol.Position) ([]types.Object, error) {
	phs, err := s.PackageHandles(ctx, f)
	if err != nil {
		return nil, err
	}

	var objs []types.Object

	// Check all the packages that the file belongs to.
	for _, ph := range phs {
		pkg, err := ph.Check(ctx)
		if err != nil {
			return nil, err
		}

		astFile, pos, err := getASTFile(pkg, f, pp)
		if err != nil {
			return nil, err
		}

		path := pathEnclosingIdent(astFile, pos)
		if len(path) == 0 {
			return nil, ErrNoIdentFound
		}

		ident := path[len(path)-1].(*ast.Ident)

		obj := pkg.GetTypesInfo().ObjectOf(ident)
		if obj == nil {
			return nil, fmt.Errorf("no object for %q", ident.Name)
		}

		objs = append(objs, obj)
	}

	return objs, nil
}

func getASTFile(pkg Package, f FileHandle, pos protocol.Position) (*ast.File, token.Pos, error) {
	pgh, err := pkg.File(f.Identity().URI)
	if err != nil {
		return nil, 0, err
	}

	file, m, _, err := pgh.Cached()
	if err != nil {
		return nil, 0, err
	}

	spn, err := m.PointSpan(pos)
	if err != nil {
		return nil, 0, err
	}

	rng, err := spn.Range(m.Converter)
	if err != nil {
		return nil, 0, err
	}

	return file, rng.Start, nil
}

// pathEnclosingIdent returns the ast path to the node that contains pos.
// It is similar to astutil.PathEnclosingInterval, but simpler, and it
// matches *ast.Ident nodes if pos is equal to node.End().
func pathEnclosingIdent(f *ast.File, pos token.Pos) []ast.Node {
	var (
		path  []ast.Node
		found bool
	)

	ast.Inspect(f, func(n ast.Node) bool {
		if found {
			return false
		}

		if n == nil {
			path = path[:len(path)-1]
			return false
		}

		switch n := n.(type) {
		case *ast.Ident:
			found = n.Pos() <= pos && pos <= n.End()
		}

		path = append(path, n)

		return !found
	})

	return path
}
