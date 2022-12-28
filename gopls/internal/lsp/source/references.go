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
	"strings"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/safetoken"
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

// References returns a list of references for a given identifier within the packages
// containing i.File. Declarations appear first in the result.
func References(ctx context.Context, snapshot Snapshot, f FileHandle, pp protocol.Position, includeDeclaration bool) ([]*ReferenceInfo, error) {
	ctx, done := event.Start(ctx, "source.References")
	defer done()

	// Is the cursor within the package name declaration?
	pgf, inPackageName, err := parsePackageNameDecl(ctx, snapshot, f, pp)
	if err != nil {
		return nil, err
	}
	if inPackageName {
		// TODO(rfindley): this is redundant with package renaming. Refactor to share logic.
		metas, err := snapshot.MetadataForFile(ctx, f.URI())
		if err != nil {
			return nil, err
		}
		if len(metas) == 0 {
			return nil, fmt.Errorf("found no package containing %s", f.URI())
		}
		targetPkg := metas[len(metas)-1] // widest package

		// Find external direct references to the package (imports).
		rdeps, err := snapshot.ReverseDependencies(ctx, targetPkg.ID, false)
		if err != nil {
			return nil, err
		}

		var refs []*ReferenceInfo
		for _, rdep := range rdeps {
			for _, uri := range rdep.CompiledGoFiles {
				fh, err := snapshot.GetFile(ctx, uri)
				if err != nil {
					return nil, err
				}
				f, err := snapshot.ParseGo(ctx, fh, ParseHeader)
				if err != nil {
					return nil, err
				}
				for _, imp := range f.File.Imports {
					if rdep.DepsByImpPath[UnquoteImportPath(imp)] == targetPkg.ID {
						refs = append(refs, &ReferenceInfo{
							Name:        pgf.File.Name.Name,
							MappedRange: NewMappedRange(f.Mapper, imp.Pos(), imp.End()),
						})
					}
				}
			}
		}

		// Find the package declaration of each file in the target package itself.
		for _, uri := range targetPkg.CompiledGoFiles {
			fh, err := snapshot.GetFile(ctx, uri)
			if err != nil {
				return nil, err
			}
			f, err := snapshot.ParseGo(ctx, fh, ParseHeader)
			if err != nil {
				return nil, err
			}
			refs = append(refs, &ReferenceInfo{
				Name:        pgf.File.Name.Name,
				MappedRange: NewMappedRange(f.Mapper, f.File.Name.Pos(), f.File.Name.End()),
			})
		}

		return refs, nil
	}

	qualifiedObjs, err := qualifiedObjsAtProtocolPos(ctx, snapshot, f.URI(), pp)
	// Don't return references for builtin types.
	if errors.Is(err, errBuiltin) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}

	refs, err := references(ctx, snapshot, qualifiedObjs, includeDeclaration, true, false)
	if err != nil {
		return nil, err
	}

	toSort := refs
	if includeDeclaration {
		toSort = refs[1:]
	}
	sort.Slice(toSort, func(i, j int) bool {
		x, y := toSort[i], toSort[j]
		if cmp := strings.Compare(string(x.URI()), string(y.URI())); cmp != 0 {
			return cmp < 0
		}
		return x.ident.Pos() < y.ident.Pos()
	})
	return refs, nil
}

// parsePackageNameDecl is a convenience function that parses and
// returns the package name declaration of file fh, and reports
// whether the position ppos lies within it.
func parsePackageNameDecl(ctx context.Context, snapshot Snapshot, fh FileHandle, ppos protocol.Position) (*ParsedGoFile, bool, error) {
	pgf, err := snapshot.ParseGo(ctx, fh, ParseHeader)
	if err != nil {
		return nil, false, err
	}
	// Careful: because we used ParseHeader,
	// Mapper.Pos(ppos) may be beyond EOF => (0, err).
	pos, _ := pgf.Mapper.Pos(ppos)
	return pgf, pgf.File.Name.Pos() <= pos && pos <= pgf.File.Name.End(), nil
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
	filename := safetoken.StartPosition(qos[0].pkg.FileSet(), pos).Filename
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
			// If obj is a package-level object, we need only search
			// among direct reverse dependencies.
			// TODO(adonovan): opt: this will still spuriously search
			// transitively for (e.g.) capitalized local variables.
			// We could do better by checking for an objectpath.
			transitive := qo.obj.Pkg().Scope().Lookup(qo.obj.Name()) != qo.obj
			rdeps, err := snapshot.ReverseDependencies(ctx, qo.pkg.ID(), transitive)
			if err != nil {
				return nil, err
			}
			ids := make([]PackageID, 0, len(rdeps))
			for _, rdep := range rdeps {
				ids = append(ids, rdep.ID)
			}
			// TODO(adonovan): opt: build a search index
			// that doesn't require type checking.
			reverseDeps, err := snapshot.TypeCheck(ctx, TypecheckFull, ids...)
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
				rng, err := posToMappedRange(pkg, ident.Pos(), ident.End())
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
