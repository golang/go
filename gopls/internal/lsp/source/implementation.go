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
	"reflect"
	"sort"
	"strings"
	"sync"

	"golang.org/x/sync/errgroup"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/safetoken"
	"golang.org/x/tools/gopls/internal/lsp/source/methodsets"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/event"
)

// This file defines the new implementation of the 'implementation'
// operator that does not require type-checker data structures for an
// unbounded number of packages.
//
// TODO(adonovan):
// - Audit to ensure robustness in face of type errors.
// - Eliminate false positives due to 'tricky' cases of the global algorithm.
// - Ensure we have test coverage of:
//      type aliases
//      nil, PkgName, Builtin (all errors)
//      any (empty result)
//      method of unnamed interface type (e.g. var x interface { f() })
//        (the global algorithm may find implementations of this type
//         but will not include it in the index.)

// Implementation returns a new sorted array of locations of
// declarations of types that implement (or are implemented by) the
// type referred to at the given position.
//
// If the position denotes a method, the computation is applied to its
// receiver type and then its corresponding methods are returned.
func Implementation(ctx context.Context, snapshot Snapshot, f FileHandle, pp protocol.Position) ([]protocol.Location, error) {
	ctx, done := event.Start(ctx, "source.Implementation")
	defer done()

	locs, err := implementations(ctx, snapshot, f, pp)
	if err != nil {
		return nil, err
	}

	// Sort and de-duplicate locations.
	sort.Slice(locs, func(i, j int) bool {
		return protocol.CompareLocation(locs[i], locs[j]) < 0
	})
	out := locs[:0]
	for _, loc := range locs {
		if len(out) == 0 || out[len(out)-1] != loc {
			out = append(out, loc)
		}
	}
	locs = out

	return locs, nil
}

func implementations(ctx context.Context, snapshot Snapshot, fh FileHandle, pp protocol.Position) ([]protocol.Location, error) {

	// Type-check the query package, find the query identifier,
	// and locate the type or method declaration it refers to.
	declPosn, err := typeDeclPosition(ctx, snapshot, fh.URI(), pp)
	if err != nil {
		return nil, err
	}

	// Type-check the declaring package (incl. variants) for use
	// by the "local" search, which uses type information to
	// enumerate all types within the package that satisfy the
	// query type, even those defined local to a function.
	declURI := span.URIFromPath(declPosn.Filename)
	declMetas, err := snapshot.MetadataForFile(ctx, declURI)
	if err != nil {
		return nil, err
	}
	RemoveIntermediateTestVariants(&declMetas)
	if len(declMetas) == 0 {
		return nil, fmt.Errorf("no packages for file %s", declURI)
	}
	ids := make([]PackageID, len(declMetas))
	for i, m := range declMetas {
		ids[i] = m.ID
	}
	localPkgs, err := snapshot.TypeCheck(ctx, ids...)
	if err != nil {
		return nil, err
	}
	// The narrowest package will do, since the local search is based
	// on position and the global search is based on fingerprint.
	// (Neither is based on object identity.)
	declPkg := localPkgs[0]
	declFile, err := declPkg.File(declURI)
	if err != nil {
		return nil, err // "can't happen"
	}

	// Find declaration of corresponding object
	// in this package based on (URI, offset).
	pos, err := safetoken.Pos(declFile.Tok, declPosn.Offset)
	if err != nil {
		return nil, err
	}
	// TODO(adonovan): simplify: use objectsAt?
	path := pathEnclosingObjNode(declFile.File, pos)
	if path == nil {
		return nil, ErrNoIdentFound // checked earlier
	}
	id, ok := path[0].(*ast.Ident)
	if !ok {
		return nil, ErrNoIdentFound // checked earlier
	}
	obj := declPkg.GetTypesInfo().ObjectOf(id) // may be nil

	// Is the selected identifier a type name or method?
	// (For methods, report the corresponding method names.)
	var queryType types.Type
	var queryMethodID string
	switch obj := obj.(type) {
	case *types.TypeName:
		queryType = obj.Type()
	case *types.Func:
		// For methods, use the receiver type, which may be anonymous.
		if recv := obj.Type().(*types.Signature).Recv(); recv != nil {
			queryType = recv.Type()
			queryMethodID = obj.Id()
		}
	}
	if queryType == nil {
		return nil, fmt.Errorf("%s is not a type or method", id.Name)
	}

	// Compute the method-set fingerprint used as a key to the global search.
	key, hasMethods := methodsets.KeyOf(queryType)
	if !hasMethods {
		// A type with no methods yields an empty result.
		// (No point reporting that every type satisfies 'any'.)
		return nil, nil
	}

	// The global search needs to look at every package in the
	// forward transitive closure of the workspace; see package
	// ./methodsets.
	//
	// For now we do all the type checking before beginning the search.
	// TODO(adonovan): opt: search in parallel topological order
	// so that we can overlap index lookup with typechecking.
	// I suspect a number of algorithms on the result of TypeCheck could
	// be optimized by being applied as soon as each package is available.
	globalMetas, err := snapshot.AllMetadata(ctx)
	if err != nil {
		return nil, err
	}
	RemoveIntermediateTestVariants(&globalMetas)
	globalIDs := make([]PackageID, 0, len(globalMetas))
	for _, m := range globalMetas {
		if m.PkgPath == declPkg.Metadata().PkgPath {
			continue // declaring package is handled by local implementation
		}
		globalIDs = append(globalIDs, m.ID)
	}
	indexes, err := snapshot.MethodSets(ctx, globalIDs...)
	if err != nil {
		return nil, fmt.Errorf("querying method sets: %v", err)
	}

	// Search local and global packages in parallel.
	var (
		group  errgroup.Group
		locsMu sync.Mutex
		locs   []protocol.Location
	)
	// local search
	for _, localPkg := range localPkgs {
		localPkg := localPkg
		group.Go(func() error {
			localLocs, err := localImplementations(ctx, snapshot, localPkg, queryType, queryMethodID)
			if err != nil {
				return err
			}
			locsMu.Lock()
			locs = append(locs, localLocs...)
			locsMu.Unlock()
			return nil
		})
	}
	// global search
	for _, index := range indexes {
		index := index
		group.Go(func() error {
			for _, res := range index.Search(key, queryMethodID) {
				loc := res.Location
				// Map offsets to protocol.Locations in parallel (may involve I/O).
				group.Go(func() error {
					ploc, err := offsetToLocation(ctx, snapshot, loc.Filename, loc.Start, loc.End)
					if err != nil {
						return err
					}
					locsMu.Lock()
					locs = append(locs, ploc)
					locsMu.Unlock()
					return nil
				})
			}
			return nil
		})
	}
	if err := group.Wait(); err != nil {
		return nil, err
	}

	return locs, nil
}

// offsetToLocation converts an offset-based position to a protocol.Location,
// which requires reading the file.
func offsetToLocation(ctx context.Context, snapshot Snapshot, filename string, start, end int) (protocol.Location, error) {
	uri := span.URIFromPath(filename)
	fh, err := snapshot.ReadFile(ctx, uri)
	if err != nil {
		return protocol.Location{}, err // cancelled, perhaps
	}
	content, err := fh.Content()
	if err != nil {
		return protocol.Location{}, err // nonexistent or deleted ("can't happen")
	}
	m := protocol.NewMapper(uri, content)
	return m.OffsetLocation(start, end)
}

// typeDeclPosition returns the position of the declaration of the
// type (or one of its methods) referred to at (uri, ppos).
func typeDeclPosition(ctx context.Context, snapshot Snapshot, uri span.URI, ppos protocol.Position) (token.Position, error) {
	var noPosn token.Position

	pkg, pgf, err := NarrowestPackageForFile(ctx, snapshot, uri)
	if err != nil {
		return noPosn, err
	}
	pos, err := pgf.PositionPos(ppos)
	if err != nil {
		return noPosn, err
	}

	// This function inherits the limitation of its predecessor in
	// requiring the selection to be an identifier (of a type or
	// method). But there's no fundamental reason why one could
	// not pose this query about any selected piece of syntax that
	// has a type and thus a method set.
	// (If LSP was more thorough about passing text selections as
	// intervals to queries, you could ask about the method set of a
	// subexpression such as x.f().)

	// TODO(adonovan): simplify: use objectsAt?
	path := pathEnclosingObjNode(pgf.File, pos)
	if path == nil {
		return noPosn, ErrNoIdentFound
	}
	id, ok := path[0].(*ast.Ident)
	if !ok {
		return noPosn, ErrNoIdentFound
	}

	// Is the object a type or method? Reject other kinds.
	obj := pkg.GetTypesInfo().Uses[id]
	if obj == nil {
		// Check uses first (unlike ObjectOf) so that T in
		// struct{T} is treated as a reference to a type,
		// not a declaration of a field.
		obj = pkg.GetTypesInfo().Defs[id]
	}
	switch obj := obj.(type) {
	case *types.TypeName:
		// ok
	case *types.Func:
		if obj.Type().(*types.Signature).Recv() == nil {
			return noPosn, fmt.Errorf("%s is a function, not a method", id.Name)
		}
	case nil:
		return noPosn, fmt.Errorf("%s denotes unknown object", id.Name)
	default:
		// e.g. *types.Var -> "var".
		kind := strings.ToLower(strings.TrimPrefix(reflect.TypeOf(obj).String(), "*types."))
		return noPosn, fmt.Errorf("%s is a %s, not a type", id.Name, kind)
	}

	declPosn := safetoken.StartPosition(pkg.FileSet(), obj.Pos())
	return declPosn, nil
}

// localImplementations searches within pkg for declarations of all
// types that are assignable to/from the query type, and returns a new
// unordered array of their locations.
//
// If methodID is non-empty, the function instead returns the location
// of each type's method (if any) of that ID.
//
// ("Local" refers to the search within the same package, but this
// function's results may include type declarations that are local to
// a function body. The global search index excludes such types
// because reliably naming such types is hard.)
func localImplementations(ctx context.Context, snapshot Snapshot, pkg Package, queryType types.Type, methodID string) ([]protocol.Location, error) {
	queryType = methodsets.EnsurePointer(queryType)

	// Scan through all type declarations in the syntax.
	var locs []protocol.Location
	var methodLocs []methodsets.Location
	for _, pgf := range pkg.CompiledGoFiles() {
		ast.Inspect(pgf.File, func(n ast.Node) bool {
			spec, ok := n.(*ast.TypeSpec)
			if !ok {
				return true // not a type declaration
			}
			def := pkg.GetTypesInfo().Defs[spec.Name]
			if def == nil {
				return true // "can't happen" for types
			}
			if def.(*types.TypeName).IsAlias() {
				return true // skip type aliases to avoid duplicate reporting
			}
			candidateType := methodsets.EnsurePointer(def.Type())

			// The historical behavior enshrined by this
			// function rejects cases where both are
			// (nontrivial) interface types?
			// That seems like useful information.
			// TODO(adonovan): UX: report I/I pairs too?
			// The same question appears in the global algorithm (methodsets).
			if !concreteImplementsIntf(candidateType, queryType) {
				return true // not assignable
			}

			// Ignore types with empty method sets.
			// (No point reporting that every type satisfies 'any'.)
			mset := types.NewMethodSet(candidateType)
			if mset.Len() == 0 {
				return true
			}

			if methodID == "" {
				// Found matching type.
				locs = append(locs, mustLocation(pgf, spec.Name))
				return true
			}

			// Find corresponding method.
			//
			// We can't use LookupFieldOrMethod because it requires
			// the methodID's types.Package, which we don't know.
			// We could recursively search pkg.Imports for it,
			// but it's easier to walk the method set.
			for i := 0; i < mset.Len(); i++ {
				method := mset.At(i).Obj()
				if method.Id() == methodID {
					posn := safetoken.StartPosition(pkg.FileSet(), method.Pos())
					methodLocs = append(methodLocs, methodsets.Location{
						Filename: posn.Filename,
						Start:    posn.Offset,
						End:      posn.Offset + len(method.Name()),
					})
					break
				}
			}
			return true
		})
	}

	// Finally convert method positions to protocol form by reading the files.
	for _, mloc := range methodLocs {
		loc, err := offsetToLocation(ctx, snapshot, mloc.Filename, mloc.Start, mloc.End)
		if err != nil {
			return nil, err
		}
		locs = append(locs, loc)
	}

	// Special case: for types that satisfy error, report builtin.go (see #59527).
	if types.Implements(queryType, errorInterfaceType) {
		loc, err := errorLocation(ctx, snapshot)
		if err != nil {
			return nil, err
		}
		locs = append(locs, loc)
	}

	return locs, nil
}

var errorInterfaceType = types.Universe.Lookup("error").Type().Underlying().(*types.Interface)

// errorLocation returns the location of the 'error' type in builtin.go.
func errorLocation(ctx context.Context, snapshot Snapshot) (protocol.Location, error) {
	pgf, err := snapshot.BuiltinFile(ctx)
	if err != nil {
		return protocol.Location{}, err
	}
	for _, decl := range pgf.File.Decls {
		if decl, ok := decl.(*ast.GenDecl); ok {
			for _, spec := range decl.Specs {
				if spec, ok := spec.(*ast.TypeSpec); ok && spec.Name.Name == "error" {
					return pgf.NodeLocation(spec.Name)
				}
			}
		}
	}
	return protocol.Location{}, fmt.Errorf("built-in error type not found")
}

// concreteImplementsIntf returns true if a is an interface type implemented by
// concrete type b, or vice versa.
func concreteImplementsIntf(a, b types.Type) bool {
	aIsIntf, bIsIntf := types.IsInterface(a), types.IsInterface(b)

	// Make sure exactly one is an interface type.
	if aIsIntf == bIsIntf {
		return false
	}

	// Rearrange if needed so "a" is the concrete type.
	if aIsIntf {
		a, b = b, a
	}

	// TODO(adonovan): this should really use GenericAssignableTo
	// to report (e.g.) "ArrayList[T] implements List[T]", but
	// GenericAssignableTo doesn't work correctly on pointers to
	// generic named types. Thus the legacy implementation and the
	// "local" part of implementations fail to report generics.
	// The global algorithm based on subsets does the right thing.
	return types.AssignableTo(a, b)
}

var (
	// TODO(adonovan): why do various RPC handlers related to
	// IncomingCalls return (nil, nil) on the protocol in response
	// to this error? That seems like a violation of the protocol.
	// Is it perhaps a workaround for VSCode behavior?
	errNoObjectFound = errors.New("no object found")
)

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
