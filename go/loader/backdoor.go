package loader

import (
	"go/ast"
	"go/build"
	"go/token"
)

// PackageLocatorFunc exposes the address of parsePackageFiles to tests.
// This is a temporary hack until we expose a proper PackageLocator interface.
func PackageLocatorFunc() *func(ctxt *build.Context, fset *token.FileSet, path string, which string) ([]*ast.File, error) {
	return &parsePackageFiles
}
