// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"go/ast"
	"sync"

	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/span"
	errors "golang.org/x/xerrors"
)

// goFile holds all of the information we know about a Go file.
type goFile struct {
	fileBase

	// mu protects all mutable state of the Go file,
	// which can be modified during type-checking.
	mu sync.Mutex

	// missingImports is the set of unresolved imports for this package.
	// It contains any packages with `go list` errors.
	missingImports map[packagePath]struct{}

	imports []*ast.ImportSpec
}

type packageKey struct {
	id   packageID
	mode source.ParseMode
}

func (f *goFile) CheckPackageHandles(ctx context.Context) (cphs []source.CheckPackageHandle, err error) {
	ctx = telemetry.File.With(ctx, f.URI())
	fh := f.Handle(ctx)

	cphs = f.isDirty(ctx, fh)
	if len(cphs) == 0 {
		cphs, err = f.view.loadParseTypecheck(ctx, f, fh)
		if err != nil {
			return nil, err
		}
	}
	if len(cphs) == 0 {
		return nil, errors.Errorf("no CheckPackageHandles for %s", f.URI())
	}
	return cphs, nil
}

func (v *view) GetActiveReverseDeps(ctx context.Context, uri span.URI) (results []source.CheckPackageHandle) {
	var (
		rdeps = v.reverseDependencies(ctx, uri)
		files = v.openFiles(ctx, rdeps)
		seen  = make(map[span.URI]struct{})
	)
	for _, f := range files {
		if _, ok := seen[f.URI()]; ok {
			continue
		}
		gof, ok := f.(source.GoFile)
		if !ok {
			continue
		}
		cphs, err := gof.CheckPackageHandles(ctx)
		if err != nil {
			continue
		}
		cph := source.WidestCheckPackageHandle(cphs)
		for _, ph := range cph.Files() {
			seen[ph.File().Identity().URI] = struct{}{}
		}
		results = append(results, cph)
	}
	return results
}

// isDirty is true if the file needs to be type-checked.
// It assumes that the file's view's mutex is held by the caller.
func (f *goFile) isDirty(ctx context.Context, fh source.FileHandle) []source.CheckPackageHandle {
	meta, cphs := f.view.getSnapshot(f.URI())
	if len(meta) == 0 {
		return nil
	}
	var results []source.CheckPackageHandle
	for key, cph := range cphs {
		// If we're explicitly checking if a file needs to be type-checked,
		// we need it to be fully parsed.
		if key.mode != source.ParseFull {
			continue
		}
		// Check if there is a fully-parsed package to which this file belongs.
		for _, file := range cph.Files() {
			if file.File().Identity() == fh.Identity() {
				results = append(results, cph)
			}
		}
	}
	return results
}
