package cache

import (
	"context"
	"go/ast"
	"go/token"

	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

// goFile holds all of the information we know about a Go file.
type goFile struct {
	fileBase

	ast *astFile

	pkg     *pkg
	meta    *metadata
	imports []*ast.ImportSpec
}

type astFile struct {
	file      *ast.File
	err       error // parse errors
	isTrimmed bool
}

func (f *goFile) GetToken(ctx context.Context) *token.File {
	f.view.mu.Lock()
	defer f.view.mu.Unlock()

	if f.isDirty() || f.astIsTrimmed() {
		if _, err := f.view.loadParseTypecheck(ctx, f); err != nil {
			f.View().Session().Logger().Errorf(ctx, "unable to check package for %s: %v", f.URI(), err)
			return nil
		}
	}
	if unexpectedAST(ctx, f) {
		return nil
	}
	return f.token
}

func (f *goFile) GetAnyAST(ctx context.Context) *ast.File {
	f.view.mu.Lock()
	defer f.view.mu.Unlock()

	if f.isDirty() {
		if _, err := f.view.loadParseTypecheck(ctx, f); err != nil {
			f.View().Session().Logger().Errorf(ctx, "unable to check package for %s: %v", f.URI(), err)
			return nil
		}
	}
	if f.ast == nil {
		return nil
	}
	return f.ast.file
}

func (f *goFile) GetAST(ctx context.Context) *ast.File {
	f.view.mu.Lock()
	defer f.view.mu.Unlock()

	if f.isDirty() || f.astIsTrimmed() {
		if _, err := f.view.loadParseTypecheck(ctx, f); err != nil {
			f.View().Session().Logger().Errorf(ctx, "unable to check package for %s: %v", f.URI(), err)
			return nil
		}
	}
	if unexpectedAST(ctx, f) {
		return nil
	}
	return f.ast.file
}

func (f *goFile) GetPackage(ctx context.Context) source.Package {
	f.view.mu.Lock()
	defer f.view.mu.Unlock()

	if f.isDirty() || f.astIsTrimmed() {
		if errs, err := f.view.loadParseTypecheck(ctx, f); err != nil {
			f.View().Session().Logger().Errorf(ctx, "unable to check package for %s: %v", f.URI(), err)

			// Create diagnostics for errors if we are able to.
			if len(errs) > 0 {
				return &pkg{errors: errs}
			}
			return nil
		}
	}
	if unexpectedAST(ctx, f) {
		return nil
	}
	return f.pkg
}

func unexpectedAST(ctx context.Context, f *goFile) bool {
	// If the AST comes back nil, something has gone wrong.
	if f.ast == nil {
		f.View().Session().Logger().Errorf(ctx, "expected full AST for %s, returned nil", f.URI())
		return true
	}
	// If the AST comes back trimmed, something has gone wrong.
	if f.astIsTrimmed() {
		f.View().Session().Logger().Errorf(ctx, "expected full AST for %s, returned trimmed", f.URI())
		return true
	}
	return false
}

// isDirty is true if the file needs to be type-checked.
// It assumes that the file's view's mutex is held by the caller.
func (f *goFile) isDirty() bool {
	return f.meta == nil || len(f.meta.missingImports) > 0 || f.token == nil || f.ast == nil || f.pkg == nil
}

func (f *goFile) astIsTrimmed() bool {
	return f.ast != nil && f.ast.isTrimmed
}

func (f *goFile) GetActiveReverseDeps(ctx context.Context) []source.GoFile {
	pkg := f.GetPackage(ctx)
	if pkg == nil {
		return nil
	}

	f.view.mu.Lock()
	defer f.view.mu.Unlock()

	f.view.mcache.mu.Lock()
	defer f.view.mcache.mu.Unlock()

	seen := make(map[packageID]struct{}) // visited packages
	results := make(map[*goFile]struct{})
	f.view.reverseDeps(ctx, seen, results, packageID(pkg.ID()))

	var files []source.GoFile
	for rd := range results {
		if rd == nil {
			continue
		}
		// Don't return any of the active files in this package.
		if rd.pkg != nil && rd.pkg == pkg {
			continue
		}
		files = append(files, rd)
	}
	return files
}

func (v *view) reverseDeps(ctx context.Context, seen map[packageID]struct{}, results map[*goFile]struct{}, id packageID) {
	if _, ok := seen[id]; ok {
		return
	}
	seen[id] = struct{}{}
	m, ok := v.mcache.packages[id]
	if !ok {
		return
	}
	for _, filename := range m.files {
		uri := span.FileURI(filename)
		if f, err := v.getFile(uri); err == nil && v.session.IsOpen(uri) {
			results[f.(*goFile)] = struct{}{}
		}
	}
	for parentID := range m.parents {
		v.reverseDeps(ctx, seen, results, parentID)
	}
}
