// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"
	"go/ast"
	"go/types"

	"golang.org/x/tools/internal/span"
)

// ReferenceInfo holds information about reference to an identifier in Go source.
type ReferenceInfo struct {
	Name  string
	Range span.Range
	ident *ast.Ident
	obj   types.Object
}

// References returns a list of references for a given identifier within a package.
func (i *IdentifierInfo) References(ctx context.Context) ([]*ReferenceInfo, error) {
	pkg := i.File.GetPackage(ctx)
	if pkg == nil || pkg.IsIllTyped() {
		return nil, fmt.Errorf("package for %s is ill typed", i.File.URI())
	}
	pkgInfo := pkg.GetTypesInfo()
	if pkgInfo == nil {
		return nil, fmt.Errorf("package %s has no types info", pkg.PkgPath())
	}

	// If the object declaration is nil, assume it is an import spec and do not look for references.
	if i.decl.obj == nil {
		return []*ReferenceInfo{}, nil
	}

	var references []*ReferenceInfo

	if i.decl.wasImplicit {
		// The definition is implicit, so we must add it separately.
		// This occurs when the variable is declared in a type switch statement
		// or is an implicit package name.
		references = append(references, &ReferenceInfo{
			Name:  i.decl.obj.Name(),
			Range: i.decl.rng,
			obj:   i.decl.obj,
		})
	}

	for ident, obj := range pkgInfo.Defs {
		if obj == nil || obj.Pos() != i.decl.obj.Pos() {
			continue
		}
		references = append(references, &ReferenceInfo{
			Name:  ident.Name,
			Range: span.NewRange(i.File.FileSet(), ident.Pos(), ident.End()),
			ident: ident,
			obj:   obj,
		})
	}

	for ident, obj := range pkgInfo.Uses {
		if obj == nil || obj.Pos() != i.decl.obj.Pos() {
			continue
		}
		references = append(references, &ReferenceInfo{
			Name:  ident.Name,
			Range: span.NewRange(i.File.FileSet(), ident.Pos(), ident.End()),
			ident: ident,
			obj:   obj,
		})
	}

	return references, nil
}
