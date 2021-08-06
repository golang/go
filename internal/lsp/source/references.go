// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"go/ast"
	"go/token"
	"go/types"
	"sort"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/span"
	errors "golang.org/x/xerrors"
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

// References returns a list of references for a given identifier within the packages
// containing i.File. Declarations appear first in the result.
func References(ctx context.Context, s Snapshot, f FileHandle, pp protocol.Position, includeDeclaration bool) ([]*ReferenceInfo, error) {
	ctx, done := event.Start(ctx, "source.References")
	defer done()

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
func references(ctx context.Context, snapshot Snapshot, qos []qualifiedObject, includeDeclaration, includeInterfaceRefs, includeEmbeddedRefs bool) ([]*ReferenceInfo, error) {
	var (
		references []*ReferenceInfo
		seen       = make(map[token.Pos]bool)
	)

	filename := snapshot.FileSet().Position(qos[0].obj.Pos()).Filename
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
				if obj != qo.obj {
					// If ident is not a use of qo.obj, skip it, with one exception: uses
					// of an embedded field can be considered references of the embedded
					// type name.
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
				if seen[ident.Pos()] {
					continue
				}
				seen[ident.Pos()] = true
				rng, err := posToMappedRange(snapshot, pkg, ident.Pos(), ident.End())
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
