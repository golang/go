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
			return nil, err
		}
		pr, err := rng.Range()
		if err != nil {
			return nil, err
		}
		locations = append(locations, protocol.Location{
			URI:   protocol.URIFromSpanURI(rng.URI()),
			Range: pr,
		})
	}
	return locations, nil
}

var ErrNotAType = errors.New("not a type name or method")

// implementations returns the concrete implementations of the specified
// interface, or the interfaces implemented by the specified concrete type.
func implementations(ctx context.Context, s Snapshot, f FileHandle, pp protocol.Position) ([]qualifiedObject, error) {
	var (
		impls []qualifiedObject
		seen  = make(map[token.Position]bool)
		fset  = s.View().Session().Cache().FileSet()
	)

	qos, err := qualifiedObjsAtProtocolPos(ctx, s, f, pp)
	if err != nil {
		return nil, err
	}

	for _, qo := range qos {
		var (
			queryType   types.Type
			queryMethod *types.Func
		)

		switch obj := qo.obj.(type) {
		case *types.Func:
			queryMethod = obj
			if recv := obj.Type().(*types.Signature).Recv(); recv != nil {
				queryType = ensurePointer(recv.Type())
			}
		case *types.TypeName:
			queryType = ensurePointer(obj.Type())
		}

		if queryType == nil {
			return nil, ErrNotAType
		}

		if types.NewMethodSet(queryType).Len() == 0 {
			return nil, nil
		}

		// Find all named types, even local types (which can have methods
		// due to promotion).
		var (
			allNamed []*types.Named
			pkgs     = make(map[*types.Package]Package)
		)
		knownPkgs, err := s.KnownPackages(ctx)
		if err != nil {
			return nil, err
		}
		for _, ph := range knownPkgs {
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
				if named, ok := obj.Type().(*types.Named); ok {
					allNamed = append(allNamed, named)
				}
			}
		}

		// Find all the named types that match our query.
		for _, named := range allNamed {
			var (
				candObj  types.Object = named.Obj()
				candType              = ensurePointer(named)
			)

			if !concreteImplementsIntf(candType, queryType) {
				continue
			}

			ms := types.NewMethodSet(candType)
			if ms.Len() == 0 {
				// Skip empty interfaces.
				continue
			}

			// If client queried a method, look up corresponding candType method.
			if queryMethod != nil {
				sel := ms.Lookup(queryMethod.Pkg(), queryMethod.Name())
				if sel == nil {
					continue
				}
				candObj = sel.Obj()
			}

			pos := fset.Position(candObj.Pos())
			if candObj == queryMethod || seen[pos] {
				continue
			}

			seen[pos] = true

			impls = append(impls, qualifiedObject{
				obj: candObj,
				pkg: pkgs[candObj.Pkg()],
			})
		}
	}

	return impls, nil
}

// concreteImplementsIntf returns true if a is an interface type implemented by
// concrete type b, or vice versa.
func concreteImplementsIntf(a, b types.Type) bool {
	aIsIntf, bIsIntf := isInterface(a), isInterface(b)

	// Make sure exactly one is an interface type.
	if aIsIntf == bIsIntf {
		return false
	}

	// Rearrange if needed so "a" is the concrete type.
	if aIsIntf {
		a, b = b, a
	}

	return types.AssignableTo(a, b)
}

// ensurePointer wraps T in a *types.Pointer if T is a named, non-interface
// type. This is useful to make sure you consider a named type's full method
// set.
func ensurePointer(T types.Type) types.Type {
	if _, ok := T.(*types.Named); ok && !isInterface(T) {
		return types.NewPointer(T)
	}

	return T
}

type qualifiedObject struct {
	obj types.Object

	// pkg is the Package that contains obj's definition.
	pkg Package

	// node is the *ast.Ident or *ast.ImportSpec we followed to find obj, if any.
	node ast.Node

	// sourcePkg is the Package that contains node, if any.
	sourcePkg Package
}

// qualifiedObjsAtProtocolPos returns info for all the type.Objects
// referenced at the given position. An object will be returned for
// every package that the file belongs to.
func qualifiedObjsAtProtocolPos(ctx context.Context, s Snapshot, f FileHandle, pp protocol.Position) ([]qualifiedObject, error) {
	phs, err := s.PackageHandles(ctx, f)
	if err != nil {
		return nil, err
	}

	var qualifiedObjs []qualifiedObject

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

		path := pathEnclosingObjNode(astFile, pos)
		if path == nil {
			return nil, ErrNoIdentFound
		}

		var objs []types.Object
		switch leaf := path[0].(type) {
		case *ast.Ident:
			// If leaf represents an implicit type switch object or the type
			// switch "assign" variable, expand to all of the type switch's
			// implicit objects.
			if implicits := typeSwitchImplicits(pkg, path); len(implicits) > 0 {
				objs = append(objs, implicits...)
			} else {
				obj := pkg.GetTypesInfo().ObjectOf(leaf)
				if obj == nil {
					return nil, fmt.Errorf("no object for %q", leaf.Name)
				}
				objs = append(objs, obj)
			}
		case *ast.ImportSpec:
			// Look up the implicit *types.PkgName.
			obj := pkg.GetTypesInfo().Implicits[leaf]
			if obj == nil {
				return nil, fmt.Errorf("no object for import %q", importPath(leaf))
			}
			objs = append(objs, obj)
		}

		pkgs := make(map[*types.Package]Package)
		pkgs[pkg.GetTypes()] = pkg
		for _, imp := range pkg.Imports() {
			pkgs[imp.GetTypes()] = imp
		}

		for _, obj := range objs {
			qualifiedObjs = append(qualifiedObjs, qualifiedObject{
				obj:       obj,
				pkg:       pkgs[obj.Pkg()],
				sourcePkg: pkg,
				node:      path[0],
			})
		}
	}

	return qualifiedObjs, nil
}

func getASTFile(pkg Package, f FileHandle, pos protocol.Position) (*ast.File, token.Pos, error) {
	pgh, err := pkg.File(f.Identity().URI)
	if err != nil {
		return nil, 0, err
	}

	file, _, m, _, err := pgh.Cached()
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

// pathEnclosingObjNode returns the AST path to the object-defining
// node associated with pos. "Object-defining" means either an
// *ast.Ident mapped directly to a types.Object or an ast.Node mapped
// implicitly to a types.Object.
func pathEnclosingObjNode(f *ast.File, pos token.Pos) []ast.Node {
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

		path = append(path, n)

		switch n := n.(type) {
		case *ast.Ident:
			// Include the position directly after identifier. This handles
			// the common case where the cursor is right after the
			// identifier the user is currently typing. Previously we
			// handled this by calling astutil.PathEnclosingInterval twice,
			// once for "pos" and once for "pos-1".
			found = n.Pos() <= pos && pos <= n.End()
		case *ast.ImportSpec:
			if n.Path.Pos() <= pos && pos < n.Path.End() {
				found = true
				// If import spec has a name, add name to path even though
				// position isn't in the name.
				if n.Name != nil {
					path = append(path, n.Name)
				}
			}
		case *ast.StarExpr:
			// Follow star expressions to the inner identifier.
			if pos == n.Star {
				pos = n.X.Pos()
			}
		case *ast.SelectorExpr:
			// If pos is on the ".", move it into the selector.
			if pos == n.X.End() {
				pos = n.Sel.Pos()
			}
		}

		return !found
	})

	if len(path) == 0 {
		return nil
	}

	// Reverse path so leaf is first element.
	for i := 0; i < len(path)/2; i++ {
		path[i], path[len(path)-1-i] = path[len(path)-1-i], path[i]
	}

	return path
}
