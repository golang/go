// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"go/ast"
	"go/token"
	"go/types"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/telemetry/trace"
)

// ReferenceInfo holds information about reference to an identifier in Go source.
type ReferenceInfo struct {
	Name string
	mappedRange
	ident         *ast.Ident
	obj           types.Object
	pkg           Package
	isDeclaration bool
}

// References returns a list of references for a given identifier within the packages
// containing i.File. Declarations appear first in the result.
func References(ctx context.Context, s Snapshot, f FileHandle, pp protocol.Position, includeDeclaration bool) ([]*ReferenceInfo, error) {
	ctx, done := trace.StartSpan(ctx, "source.References")
	defer done()

	qualifiedObjs, err := qualifiedObjsAtProtocolPos(ctx, s, f, pp)
	if err != nil {
		return nil, err
	}

	var (
		references []*ReferenceInfo
		seen       = make(map[token.Position]bool)
		fset       = s.View().Session().Cache().FileSet()
	)

	// Make sure declaration is the first item in the response.
	if includeDeclaration {
		rng, err := objToMappedRange(s.View(), qualifiedObjs[0].pkg, qualifiedObjs[0].obj)
		if err != nil {
			return nil, err
		}

		ident, _ := qualifiedObjs[0].node.(*ast.Ident)
		references = append(references, &ReferenceInfo{
			mappedRange:   rng,
			Name:          qualifiedObjs[0].obj.Name(),
			ident:         ident,
			obj:           qualifiedObjs[0].obj,
			pkg:           qualifiedObjs[0].pkg,
			isDeclaration: true,
		})
	}

	for _, qo := range qualifiedObjs {
		var searchPkgs []Package

		// Only search dependents if the object is exported.
		if qo.obj.Exported() {
			reverseDeps, err := s.GetReverseDependencies(ctx, qo.pkg.ID())
			if err != nil {
				return nil, err
			}

			for _, ph := range reverseDeps {
				pkg, err := ph.Check(ctx)
				if err != nil {
					return nil, err
				}
				searchPkgs = append(searchPkgs, pkg)
			}
		}

		// Add the package in which the identifier is declared.
		searchPkgs = append(searchPkgs, qo.pkg)

		for _, pkg := range searchPkgs {
			for ident, obj := range pkg.GetTypesInfo().Uses {
				if obj != qo.obj {
					continue
				}

				pos := fset.Position(ident.Pos())
				if seen[pos] {
					continue
				}
				seen[pos] = true

				rng, err := posToMappedRange(s.View(), pkg, ident.Pos(), ident.End())
				if err != nil {
					return nil, err
				}
				references = append(references, &ReferenceInfo{
					Name:        ident.Name,
					ident:       ident,
					pkg:         pkg,
					obj:         obj,
					mappedRange: rng,
				})
			}
		}
	}

	return references, nil
}
