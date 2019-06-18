package source

import (
	"context"
	"fmt"
	"go/token"
	"go/types"

	"golang.org/x/tools/internal/span"
)

// Rename returns a map of TextEdits for each file modified when renaming a given identifier within a package.
func Rename(ctx context.Context, view View, f GoFile, pos token.Pos, newName string) (map[span.URI][]TextEdit, error) {
	pkg := f.GetPackage(ctx)
	if pkg == nil || pkg.IsIllTyped() {
		return nil, fmt.Errorf("package for %s is ill typed", f.URI())
	}

	// Get the identifier to rename.
	ident, err := Identifier(ctx, view, f, pos)
	if err != nil {
		return nil, err
	}
	if ident.Name == newName {
		return nil, fmt.Errorf("old and new names are the same: %s", newName)
	}

	// Do not rename identifiers declared in another package.
	if pkg.GetTypes() != ident.decl.obj.Pkg() {
		return nil, fmt.Errorf("failed to rename because %q is declared in package %q", ident.Name, ident.decl.obj.Pkg().Name())
	}

	// TODO(suzmue): Support renaming of imported packages.
	if _, ok := ident.decl.obj.(*types.PkgName); ok {
		return nil, fmt.Errorf("renaming imported package %s not supported", ident.Name)
	}

	// TODO(suzmue): Check that renaming ident is ok.
	refs, err := ident.References(ctx)
	if err != nil {
		return nil, err
	}

	changes := make(map[span.URI][]TextEdit)
	for _, ref := range refs {
		refSpan, err := ref.Range.Span()
		if err != nil {
			return nil, err
		}

		edit := TextEdit{
			Span:    refSpan,
			NewText: newName,
		}
		changes[refSpan.URI()] = append(changes[refSpan.URI()], edit)
	}

	return changes, nil
}
