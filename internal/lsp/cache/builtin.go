// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"go/ast"
	"strings"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
	errors "golang.org/x/xerrors"
)

type builtinPkg struct {
	pkg   *ast.Package
	files []source.ParseGoHandle
}

func (v *view) LookupBuiltin(name string) (*ast.Object, error) {
	if v.builtin == nil || v.builtin.pkg == nil || v.builtin.pkg.Scope == nil {
		return nil, errors.Errorf("no builtin package")
	}
	astObj := v.builtin.pkg.Scope.Lookup(name)
	if astObj == nil {
		return nil, errors.Errorf("no builtin object for %s", name)
	}
	return astObj, nil
}

func (b *builtinPkg) CompiledGoFiles() []source.ParseGoHandle {
	return b.files
}

// buildBuiltinPkg builds the view's builtin package.
// It assumes that the view is not active yet,
// i.e. it has not been added to the session's list of views.
func (v *view) buildBuiltinPackage(ctx context.Context) error {
	if v.builtin != nil {
		return errors.Errorf("rebuilding builtin package")
	}
	cfg := v.Config(ctx)
	pkgs, err := packages.Load(cfg, "builtin")
	// If the error is related to a go.mod parse error, we want to continue loading.
	if err != nil && strings.Contains(err.Error(), ".mod:") {
		return nil
	}
	if err != nil {
		return err
	}
	if len(pkgs) != 1 {
		return errors.Errorf("expected 1 (got %v) packages for builtin", len(pkgs))
	}
	files := make(map[string]*ast.File)
	var pghs []source.ParseGoHandle
	for _, filename := range pkgs[0].GoFiles {
		fh := v.session.GetFile(span.FileURI(filename))
		pgh := v.session.cache.ParseGoHandle(fh, source.ParseFull)
		pghs = append(pghs, pgh)
		file, _, _, err := pgh.Parse(ctx)
		if err != nil {
			return err
		}
		files[filename] = file

		v.ignoredURIsMu.Lock()
		v.ignoredURIs[span.NewURI(filename)] = struct{}{}
		v.ignoredURIsMu.Unlock()
	}
	pkg, err := ast.NewPackage(cfg.Fset, files, nil, nil)
	if err != nil {
		return err
	}
	v.builtin = &builtinPkg{
		files: pghs,
		pkg:   pkg,
	}
	return nil
}
