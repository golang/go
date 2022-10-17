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
	"strconv"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/bug"
	"golang.org/x/tools/internal/event"
)

// ReferenceInfo holds information about reference to an identifier in Go source.
type ReferenceInfo struct {
	Name string
	MappedRange
	ident         *ast.Ident
	obj           types.Object
	pkg           Package
	isDeclaration bool
}

// isInPackageName reports whether the file's package name surrounds the
// given position pp (e.g. "foo" surrounds the cursor in "package foo").
func isInPackageName(ctx context.Context, s Snapshot, f FileHandle, pgf *ParsedGoFile, pp protocol.Position) (bool, error) {
	// Find position of the package name declaration
	cursorPos, err := pgf.Mapper.Pos(pp)
	if err != nil {
		return false, err
	}

	return pgf.File.Name.Pos() <= cursorPos && cursorPos <= pgf.File.Name.End(), nil
}

// References returns a list of references for a given identifier within the packages
// containing i.File. Declarations appear first in the result.
func References(ctx context.Context, s Snapshot, f FileHandle, pp protocol.Position, includeDeclaration bool) ([]*ReferenceInfo, error) {
	ctx, done := event.Start(ctx, "source.References")
	defer done()

	// Find position of the package name declaration
	pgf, err := s.ParseGo(ctx, f, ParseFull)
	if err != nil {
		return nil, err
	}

	packageName := pgf.File.Name.Name // from package decl
	inPackageName, err := isInPackageName(ctx, s, f, pgf, pp)
	if err != nil {
		return nil, err
	}

	if inPackageName {
		// TODO(rfindley): this is inaccurate, excluding test variants, and
		// redundant with package renaming. Refactor to share logic.
		renamingPkg, err := s.PackageForFile(ctx, f.URI(), TypecheckWorkspace, NarrowestPackage)
		if err != nil {
			return nil, err
		}

		// Find external references to the package.
		rdeps, err := s.GetReverseDependencies(ctx, renamingPkg.ID())
		if err != nil {
			return nil, err
		}
		var refs []*ReferenceInfo
		for _, dep := range rdeps {
			for _, f := range dep.CompiledGoFiles() {
				for _, imp := range f.File.Imports {
					if path, err := strconv.Unquote(imp.Path.Value); err == nil && path == renamingPkg.PkgPath() {
						refs = append(refs, &ReferenceInfo{
							Name:        packageName,
							MappedRange: NewMappedRange(f.Tok, f.Mapper, imp.Pos(), imp.End()),
						})
					}
				}
			}
		}

		// Find internal references to the package within the package itself
		for _, f := range renamingPkg.CompiledGoFiles() {
			refs = append(refs, &ReferenceInfo{
				Name:        packageName,
				MappedRange: NewMappedRange(f.Tok, f.Mapper, f.File.Name.Pos(), f.File.Name.End()),
			})
		}

		return refs, nil
	}

	qualifiedObjs, err := qualifiedObjsAtProtocolPos(ctx, s, f.URI(), pp)
	// Don't return references for builtin types.
	if errors.Is(err, errBuiltin) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}

	refs, err := references(ctx, s, qualifiedObjs, includeDeclaration, true, false)
	if err != nil {
		return nil, err
	}

	toSort := refs
	if includeDeclaration {
		toSort = refs[1:]
	}
	sort.Slice(toSort, func(i, j int) bool {
		x := CompareURI(toSort[i].URI(), toSort[j].URI())
		if x == 0 {
			return toSort[i].ident.Pos() < toSort[j].ident.Pos()
		}
		return x < 0
	})
	return refs, nil
}

// references is a helper function to avoid recomputing qualifiedObjsAtProtocolPos.
// The first element of qos is considered to be the declaration;
// if isDeclaration, the first result is an extra item for it.
// Only the definition-related fields of qualifiedObject are used.
// (Arguably it should accept a smaller data type.)
func references(ctx context.Context, snapshot Snapshot, qos []qualifiedObject, includeDeclaration, includeInterfaceRefs, includeEmbeddedRefs bool) ([]*ReferenceInfo, error) {
	var (
		references []*ReferenceInfo
		seen       = make(map[positionKey]bool)
	)

	pos := qos[0].obj.Pos()
	if pos == token.NoPos {
		return nil, fmt.Errorf("no position for %s", qos[0].obj) // e.g. error.Error
	}
	// Inv: qos[0].pkg != nil, since Pos is valid.
	// Inv: qos[*].pkg != nil, since all qos are logically the same declaration.
	filename := snapshot.FileSet().Position(pos).Filename
	pgf, err := qos[0].pkg.File(span.URIFromPath(filename))
	if err != nil {
		return nil, err
	}
	declIdent, err := findIdentifier(ctx, snapshot, qos[0].pkg, pgf, qos[0].obj.Pos())
	if err != nil {
		return nil, err
	}
	// Make sure declaration is the first item in the response.
	if includeDeclaration {
		references = append(references, &ReferenceInfo{
			MappedRange:   declIdent.MappedRange,
			Name:          qos[0].obj.Name(),
			ident:         declIdent.ident,
			obj:           qos[0].obj,
			pkg:           declIdent.pkg,
			isDeclaration: true,
		})
	}

	for _, qo := range qos {
		var searchPkgs []Package

		// Only search dependents if the object is exported.
		if qo.obj.Exported() {
			reverseDeps, err := snapshot.GetReverseDependencies(ctx, qo.pkg.ID())
			if err != nil {
				return nil, err
			}
			searchPkgs = append(searchPkgs, reverseDeps...)
		}
		// Add the package in which the identifier is declared.
		searchPkgs = append(searchPkgs, qo.pkg)
		for _, pkg := range searchPkgs {
			for ident, obj := range pkg.GetTypesInfo().Uses {
				// For instantiated objects (as in methods or fields on instantiated
				// types), we may not have pointer-identical objects but still want to
				// consider them references.
				if !equalOrigin(obj, qo.obj) {
					// If ident is not a use of qo.obj, skip it, with one exception:
					// uses of an embedded field can be considered references of the
					// embedded type name
					if !includeEmbeddedRefs {
						continue
					}
					v, ok := obj.(*types.Var)
					if !ok || !v.Embedded() {
						continue
					}
					named, ok := v.Type().(*types.Named)
					if !ok || named.Obj() != qo.obj {
						continue
					}
				}
				key, found := packagePositionKey(pkg, ident.Pos())
				if !found {
					bug.Reportf("ident %v (pos: %v) not found in package %v", ident.Name, ident.Pos(), pkg.Name())
					continue
				}
				if seen[key] {
					continue
				}
				seen[key] = true
				rng, err := posToMappedRange(snapshot.FileSet(), pkg, ident.Pos(), ident.End())
				if err != nil {
					return nil, err
				}
				references = append(references, &ReferenceInfo{
					Name:        ident.Name,
					ident:       ident,
					pkg:         pkg,
					obj:         obj,
					MappedRange: rng,
				})
			}
		}
	}

	// When searching on type name, don't include interface references -- they
	// would be things like all references to Stringer for any type that
	// happened to have a String method.
	_, isType := declIdent.Declaration.obj.(*types.TypeName)
	if includeInterfaceRefs && !isType {
		// TODO(adonovan): opt: don't go back into the position domain:
		// we have complete type information already.
		declRange, err := declIdent.Range()
		if err != nil {
			return nil, err
		}
		fh, err := snapshot.GetFile(ctx, declIdent.URI())
		if err != nil {
			return nil, err
		}
		interfaceRefs, err := interfaceReferences(ctx, snapshot, fh, declRange.Start)
		if err != nil {
			return nil, err
		}
		references = append(references, interfaceRefs...)
	}

	return references, nil
}

// equalOrigin reports whether obj1 and obj2 have equivalent origin object.
// This may be the case even if obj1 != obj2, if one or both of them is
// instantiated.
func equalOrigin(obj1, obj2 types.Object) bool {
	return obj1.Pkg() == obj2.Pkg() && obj1.Pos() == obj2.Pos() && obj1.Name() == obj2.Name()
}

// interfaceReferences returns the references to the interfaces implemented by
// the type or method at the given position.
func interfaceReferences(ctx context.Context, s Snapshot, f FileHandle, pp protocol.Position) ([]*ReferenceInfo, error) {
	implementations, err := implementations(ctx, s, f, pp)
	if err != nil {
		if errors.Is(err, ErrNotAType) {
			return nil, nil
		}
		return nil, err
	}

	// Make a separate call to references() for each element
	// since it treats the first qualifiedObject as a definition.
	var refs []*ReferenceInfo
	for _, impl := range implementations {
		implRefs, err := references(ctx, s, []qualifiedObject{impl}, false, false, false)
		if err != nil {
			return nil, err
		}
		refs = append(refs, implRefs...)
	}
	return refs, nil
}
