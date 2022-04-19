// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"errors"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"sort"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/span"
)

func Implementation(ctx context.Context, snapshot Snapshot, f FileHandle, pp protocol.Position) ([]protocol.Location, error) {
	ctx, done := event.Start(ctx, "source.Implementation")
	defer done()

	impls, err := implementations(ctx, snapshot, f, pp)
	if err != nil {
		return nil, err
	}
	var locations []protocol.Location
	for _, impl := range impls {
		if impl.pkg == nil || len(impl.pkg.CompiledGoFiles()) == 0 {
			continue
		}
		rng, err := objToMappedRange(snapshot, impl.pkg, impl.obj)
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
	sort.Slice(locations, func(i, j int) bool {
		li, lj := locations[i], locations[j]
		if li.URI == lj.URI {
			return protocol.CompareRange(li.Range, lj.Range) < 0
		}
		return li.URI < lj.URI
	})
	return locations, nil
}

var ErrNotAType = errors.New("not a type name or method")

// implementations returns the concrete implementations of the specified
// interface, or the interfaces implemented by the specified concrete type.
func implementations(ctx context.Context, s Snapshot, f FileHandle, pp protocol.Position) ([]qualifiedObject, error) {
	var (
		impls []qualifiedObject
		seen  = make(map[token.Position]bool)
		fset  = s.FileSet()
	)

	qos, err := qualifiedObjsAtProtocolPos(ctx, s, f.URI(), pp)
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
		for _, pkg := range knownPkgs {
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
	aIsIntf, bIsIntf := IsInterface(a), IsInterface(b)

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
	if _, ok := T.(*types.Named); ok && !IsInterface(T) {
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

var (
	errBuiltin       = errors.New("builtin object")
	errNoObjectFound = errors.New("no object found")
)

// qualifiedObjsAtProtocolPos returns info for all the type.Objects
// referenced at the given position. An object will be returned for
// every package that the file belongs to, in every typechecking mode
// applicable.
func qualifiedObjsAtProtocolPos(ctx context.Context, s Snapshot, uri span.URI, pp protocol.Position) ([]qualifiedObject, error) {
	pkgs, err := s.PackagesForFile(ctx, uri, TypecheckAll, false)
	if err != nil {
		return nil, err
	}
	if len(pkgs) == 0 {
		return nil, errNoObjectFound
	}
	pkg := pkgs[0]
	pgf, err := pkg.File(uri)
	if err != nil {
		return nil, err
	}
	spn, err := pgf.Mapper.PointSpan(pp)
	if err != nil {
		return nil, err
	}
	rng, err := spn.Range(pgf.Mapper.Converter)
	if err != nil {
		return nil, err
	}
	offset, err := Offset(pgf.Tok, rng.Start)
	if err != nil {
		return nil, err
	}
	return qualifiedObjsAtLocation(ctx, s, objSearchKey{uri, offset}, map[objSearchKey]bool{})
}

type objSearchKey struct {
	uri    span.URI
	offset int
}

// qualifiedObjsAtLocation finds all objects referenced at offset in uri, across
// all packages in the snapshot.
func qualifiedObjsAtLocation(ctx context.Context, s Snapshot, key objSearchKey, seen map[objSearchKey]bool) ([]qualifiedObject, error) {
	if seen[key] {
		return nil, nil
	}
	seen[key] = true

	// We search for referenced objects starting with all packages containing the
	// current location, and then repeating the search for every distinct object
	// location discovered.
	//
	// In the common case, there should be at most one additional location to
	// consider: the definition of the object referenced by the location. But we
	// try to be comprehensive in case we ever support variations on build
	// constraints.

	pkgs, err := s.PackagesForFile(ctx, key.uri, TypecheckAll, false)
	if err != nil {
		return nil, err
	}

	// report objects in the order we encounter them. This ensures that the first
	// result is at the cursor...
	var qualifiedObjs []qualifiedObject
	// ...but avoid duplicates.
	seenObjs := map[types.Object]bool{}

	for _, searchpkg := range pkgs {
		pgf, err := searchpkg.File(key.uri)
		if err != nil {
			return nil, err
		}
		pos := pgf.Tok.Pos(key.offset)
		path := pathEnclosingObjNode(pgf.File, pos)
		if path == nil {
			continue
		}
		var objs []types.Object
		switch leaf := path[0].(type) {
		case *ast.Ident:
			// If leaf represents an implicit type switch object or the type
			// switch "assign" variable, expand to all of the type switch's
			// implicit objects.
			if implicits, _ := typeSwitchImplicits(searchpkg, path); len(implicits) > 0 {
				objs = append(objs, implicits...)
			} else {
				obj := searchpkg.GetTypesInfo().ObjectOf(leaf)
				if obj == nil {
					return nil, fmt.Errorf("%w for %q", errNoObjectFound, leaf.Name)
				}
				objs = append(objs, obj)
			}
		case *ast.ImportSpec:
			// Look up the implicit *types.PkgName.
			obj := searchpkg.GetTypesInfo().Implicits[leaf]
			if obj == nil {
				return nil, fmt.Errorf("%w for import %q", errNoObjectFound, ImportPath(leaf))
			}
			objs = append(objs, obj)
		}
		// Get all of the transitive dependencies of the search package.
		pkgs := make(map[*types.Package]Package)
		var addPkg func(pkg Package)
		addPkg = func(pkg Package) {
			pkgs[pkg.GetTypes()] = pkg
			for _, imp := range pkg.Imports() {
				if _, ok := pkgs[imp.GetTypes()]; !ok {
					addPkg(imp)
				}
			}
		}
		addPkg(searchpkg)
		for _, obj := range objs {
			if obj.Parent() == types.Universe {
				return nil, fmt.Errorf("%q: %w", obj.Name(), errBuiltin)
			}
			pkg, ok := pkgs[obj.Pkg()]
			if !ok {
				event.Error(ctx, fmt.Sprintf("no package for obj %s: %v", obj, obj.Pkg()), err)
				continue
			}
			qualifiedObjs = append(qualifiedObjs, qualifiedObject{
				obj:       obj,
				pkg:       pkg,
				sourcePkg: searchpkg,
				node:      path[0],
			})
			seenObjs[obj] = true

			// If the qualified object is in another file (or more likely, another
			// package), it's possible that there is another copy of it in a package
			// that we haven't searched, e.g. a test variant. See golang/go#47564.
			//
			// In order to be sure we've considered all packages, call
			// qualifiedObjsAtLocation recursively for all locations we encounter. We
			// could probably be more precise here, only continuing the search if obj
			// is in another package, but this should be good enough to find all
			// uses.

			pos := obj.Pos()
			var uri span.URI
			offset := -1
			for _, pgf := range pkg.CompiledGoFiles() {
				if pgf.Tok.Base() <= int(pos) && int(pos) <= pgf.Tok.Base()+pgf.Tok.Size() {
					var err error
					offset, err = Offset(pgf.Tok, pos)
					if err != nil {
						return nil, err
					}
					uri = pgf.URI
				}
			}
			if offset >= 0 {
				otherObjs, err := qualifiedObjsAtLocation(ctx, s, objSearchKey{uri, offset}, seen)
				if err != nil {
					return nil, err
				}
				for _, other := range otherObjs {
					if !seenObjs[other.obj] {
						qualifiedObjs = append(qualifiedObjs, other)
						seenObjs[other.obj] = true
					}
				}
			} else {
				return nil, fmt.Errorf("missing file for position of %q in %q", obj.Name(), obj.Pkg().Name())
			}
		}
	}
	// Return an error if no objects were found since callers will assume that
	// the slice has at least 1 element.
	if len(qualifiedObjs) == 0 {
		return nil, errNoObjectFound
	}
	return qualifiedObjs, nil
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
