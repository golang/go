package source

import (
	"context"
	"fmt"
	"go/ast"

	"golang.org/x/tools/internal/span"
)

// ReferenceInfo holds information about reference to an identifier in Go source.
type ReferenceInfo struct {
	Name  string
	Range span.Range
	ident *ast.Ident
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
	declObj := i.decl.obj
	if declObj == nil {
		return []*ReferenceInfo{}, nil
	}

	var references []*ReferenceInfo
	for ident, obj := range pkgInfo.Defs {
		if obj == declObj {
			references = append(references, &ReferenceInfo{
				Name:  ident.Name,
				Range: span.NewRange(i.File.FileSet(), ident.Pos(), ident.End()),
				ident: ident,
			})
		}
	}
	for ident, obj := range pkgInfo.Uses {
		if obj == declObj {
			references = append(references, &ReferenceInfo{
				Name:  ident.Name,
				Range: span.NewRange(i.File.FileSet(), ident.Pos(), ident.End()),
				ident: ident,
			})
		}
	}

	return references, nil
}
