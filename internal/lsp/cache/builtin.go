package cache

import (
	"context"
	"go/ast"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

type builtinPkg struct {
	pkg   *ast.Package
	files []source.ParseGoHandle
}

func (b *builtinPkg) Lookup(name string) *ast.Object {
	if b == nil || b.pkg == nil || b.pkg.Scope == nil {
		return nil
	}
	return b.pkg.Scope.Lookup(name)
}

func (b *builtinPkg) Files() []source.ParseGoHandle {
	return b.files
}

// buildBuiltinPkg builds the view's builtin package.
// It assumes that the view is not active yet,
// i.e. it has not been added to the session's list of views.
func (v *view) buildBuiltinPackage(ctx context.Context) error {
	cfg := v.Config(ctx)
	pkgs, err := packages.Load(cfg, "builtin")
	if err != nil {
		return err
	}
	if len(pkgs) != 1 {
		return err
	}
	pkg := pkgs[0]
	files := make(map[string]*ast.File)
	for _, filename := range pkg.GoFiles {
		fh := v.session.GetFile(span.FileURI(filename), source.Go)
		ph := v.session.cache.ParseGoHandle(fh, source.ParseFull)
		v.builtin.files = append(v.builtin.files, ph)
		file, _, _, err := ph.Parse(ctx)
		if err != nil {
			return err
		}
		files[filename] = file

		v.ignoredURIsMu.Lock()
		v.ignoredURIs[span.NewURI(filename)] = struct{}{}
		v.ignoredURIsMu.Unlock()
	}
	v.builtin.pkg, err = ast.NewPackage(cfg.Fset, files, nil, nil)
	return err
}
