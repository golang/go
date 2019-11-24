// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"go/ast"
	"go/types"

	"golang.org/x/tools/go/types/objectpath"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/telemetry/log"
	"golang.org/x/tools/internal/telemetry/trace"
	errors "golang.org/x/xerrors"
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
func (i *IdentifierInfo) References(ctx context.Context) ([]*ReferenceInfo, error) {
	ctx, done := trace.StartSpan(ctx, "source.References")
	defer done()

	var references []*ReferenceInfo

	// If the object declaration is nil, assume it is an import spec and do not look for references.
	if i.Declaration.obj == nil {
		return nil, errors.Errorf("no references for an import spec")
	}
	info := i.pkg.GetTypesInfo()
	if info == nil {
		return nil, errors.Errorf("package %s has no types info", i.pkg.PkgPath())
	}
	if i.Declaration.wasImplicit {
		// The definition is implicit, so we must add it separately.
		// This occurs when the variable is declared in a type switch statement
		// or is an implicit package name. Both implicits are local to a file.
		references = append(references, &ReferenceInfo{
			Name:          i.Declaration.obj.Name(),
			mappedRange:   i.Declaration.mappedRange,
			obj:           i.Declaration.obj,
			pkg:           i.pkg,
			isDeclaration: true,
		})
	}
	for ident, obj := range info.Defs {
		if obj == nil || !sameObj(obj, i.Declaration.obj) {
			continue
		}
		rng, err := posToMappedRange(i.Snapshot.View(), i.pkg, ident.Pos(), ident.End())
		if err != nil {
			return nil, err
		}
		// Add the declarations at the beginning of the references list.
		references = append([]*ReferenceInfo{{
			Name:          ident.Name,
			ident:         ident,
			obj:           obj,
			pkg:           i.pkg,
			isDeclaration: true,
			mappedRange:   rng,
		}}, references...)
	}
	var searchpkgs []Package
	if i.Declaration.obj.Exported() {
		// Only search all packages if the identifier is exported.
		for _, id := range i.Snapshot.GetReverseDependencies(i.pkg.ID()) {
			cph, err := i.Snapshot.PackageHandle(ctx, id)
			if err != nil {
				log.Error(ctx, "References: no CheckPackageHandle", err, telemetry.Package.Of(id))
				continue
			}
			pkg, err := cph.Check(ctx)
			if err != nil {
				log.Error(ctx, "References: no Package", err, telemetry.Package.Of(id))
				continue
			}
			searchpkgs = append(searchpkgs, pkg)
		}
	}
	// Add the package in which the identifier is declared.
	searchpkgs = append(searchpkgs, i.pkg)
	for _, pkg := range searchpkgs {
		for ident, obj := range pkg.GetTypesInfo().Uses {
			if obj == nil || !(sameObj(obj, i.Declaration.obj)) {
				continue
			}
			rng, err := posToMappedRange(i.Snapshot.View(), pkg, ident.Pos(), ident.End())
			if err != nil {
				return nil, err
			}
			references = append(references, &ReferenceInfo{
				Name:        ident.Name,
				ident:       ident,
				pkg:         i.pkg,
				obj:         obj,
				mappedRange: rng,
			})
		}
	}
	return references, nil
}

// sameObj returns true if obj is the same as declObj.
// Objects are the same if either they have they have objectpaths
// and their objectpath and package are the same; or if they don't
// have object paths and they have the same Pos and Name.
func sameObj(obj, declObj types.Object) bool {
	// TODO(suzmue): support the case where an identifier may have two different
	// declaration positions.
	if obj.Pkg() == nil || declObj.Pkg() == nil {
		if obj.Pkg() != declObj.Pkg() {
			return false
		}
	} else if obj.Pkg().Path() != declObj.Pkg().Path() {
		return false
	}
	objPath, operr := objectpath.For(obj)
	declObjPath, doperr := objectpath.For(declObj)
	if operr != nil || doperr != nil {
		return obj.Pos() == declObj.Pos() && obj.Name() == declObj.Name()
	}
	return objPath == declObjPath
}
