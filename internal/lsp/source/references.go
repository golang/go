// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"
	"go/ast"
	"go/types"

	"golang.org/x/tools/internal/lsp/telemetry/trace"
	"golang.org/x/tools/internal/span"
)

// ReferenceInfo holds information about reference to an identifier in Go source.
type ReferenceInfo struct {
	Name          string
	Range         span.Range
	ident         *ast.Ident
	obj           types.Object
	isDeclaration bool
}

// References returns a list of references for a given identifier within the packages
// containing i.File. Declarations appear first in the result.
func (i *IdentifierInfo) References(ctx context.Context) ([]*ReferenceInfo, error) {
	ctx, ts := trace.StartSpan(ctx, "source.References")
	defer ts.End()
	var references []*ReferenceInfo

	// If the object declaration is nil, assume it is an import spec and do not look for references.
	if i.decl.obj == nil {
		return nil, fmt.Errorf("no references for an import spec")
	}
	if i.decl.wasImplicit {
		// The definition is implicit, so we must add it separately.
		// This occurs when the variable is declared in a type switch statement
		// or is an implicit package name. Both implicits are local to a file.
		references = append(references, &ReferenceInfo{
			Name:          i.decl.obj.Name(),
			Range:         i.decl.rng,
			obj:           i.decl.obj,
			isDeclaration: true,
		})
	}

	pkgs := i.File.GetPackages(ctx)
	for _, pkg := range pkgs {
		if pkg == nil || pkg.IsIllTyped() {
			return nil, fmt.Errorf("package for %s is ill typed", i.File.URI())
		}
		info := pkg.GetTypesInfo()
		if info == nil {
			return nil, fmt.Errorf("package %s has no types info", pkg.PkgPath())
		}
		for ident, obj := range info.Defs {
			// TODO(suzmue): support the case where an identifier may have two different declarations.
			if obj == nil || obj.Pos() != i.decl.obj.Pos() {
				continue
			}
			// Add the declarations at the beginning of the references list.
			references = append([]*ReferenceInfo{&ReferenceInfo{
				Name:          ident.Name,
				Range:         span.NewRange(i.File.FileSet(), ident.Pos(), ident.End()),
				ident:         ident,
				obj:           obj,
				isDeclaration: true,
			}}, references...)
		}
		for ident, obj := range info.Uses {
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

	}

	return references, nil
}
