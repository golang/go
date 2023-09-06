// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"path/filepath"
	"strings"

	"golang.org/x/mod/modfile"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/span"
)

// TODO(rfindley): now that experimentalWorkspaceModule is gone, this file can
// be massively cleaned up and/or removed.

// computeWorkspaceModFiles computes the set of workspace mod files based on the
// value of go.mod, go.work, and GO111MODULE.
func computeWorkspaceModFiles(ctx context.Context, gomod, gowork span.URI, go111module go111module, fs source.FileSource) (map[span.URI]struct{}, error) {
	if go111module == off {
		return nil, nil
	}
	if gowork != "" {
		fh, err := fs.ReadFile(ctx, gowork)
		if err != nil {
			return nil, err
		}
		content, err := fh.Content()
		if err != nil {
			return nil, err
		}
		filename := gowork.Filename()
		dir := filepath.Dir(filename)
		workFile, err := modfile.ParseWork(filename, content, nil)
		if err != nil {
			return nil, fmt.Errorf("parsing go.work: %w", err)
		}
		modFiles := make(map[span.URI]struct{})
		for _, use := range workFile.Use {
			modDir := filepath.FromSlash(use.Path)
			if !filepath.IsAbs(modDir) {
				modDir = filepath.Join(dir, modDir)
			}
			modURI := span.URIFromPath(filepath.Join(modDir, "go.mod"))
			modFiles[modURI] = struct{}{}
		}
		return modFiles, nil
	}
	if gomod != "" {
		return map[span.URI]struct{}{gomod: {}}, nil
	}
	return nil, nil
}

// isGoMod reports if uri is a go.mod file.
func isGoMod(uri span.URI) bool {
	return filepath.Base(uri.Filename()) == "go.mod"
}

// isGoWork reports if uri is a go.work file.
func isGoWork(uri span.URI) bool {
	return filepath.Base(uri.Filename()) == "go.work"
}

// fileExists reports whether the file has a Content (which may be empty).
// An overlay exists even if it is not reflected in the file system.
func fileExists(fh source.FileHandle) bool {
	_, err := fh.Content()
	return err == nil
}

// errExhausted is returned by findModules if the file scan limit is reached.
var errExhausted = errors.New("exhausted")

// Limit go.mod search to 1 million files. As a point of reference,
// Kubernetes has 22K files (as of 2020-11-24).
//
// Note: per golang/go#56496, the previous limit of 1M files was too slow, at
// which point this limit was decreased to 100K.
const fileLimit = 100_000

// findModules recursively walks the root directory looking for go.mod files,
// returning the set of modules it discovers. If modLimit is non-zero,
// searching stops once modLimit modules have been found.
//
// TODO(rfindley): consider overlays.
func findModules(root span.URI, excludePath func(string) bool, modLimit int) (map[span.URI]struct{}, error) {
	// Walk the view's folder to find all modules in the view.
	modFiles := make(map[span.URI]struct{})
	searched := 0
	errDone := errors.New("done")
	err := filepath.WalkDir(root.Filename(), func(path string, info fs.DirEntry, err error) error {
		if err != nil {
			// Probably a permission error. Keep looking.
			return filepath.SkipDir
		}
		// For any path that is not the workspace folder, check if the path
		// would be ignored by the go command. Vendor directories also do not
		// contain workspace modules.
		if info.IsDir() && path != root.Filename() {
			suffix := strings.TrimPrefix(path, root.Filename())
			switch {
			case checkIgnored(suffix),
				strings.Contains(filepath.ToSlash(suffix), "/vendor/"),
				excludePath(suffix):
				return filepath.SkipDir
			}
		}
		// We're only interested in go.mod files.
		uri := span.URIFromPath(path)
		if isGoMod(uri) {
			modFiles[uri] = struct{}{}
		}
		if modLimit > 0 && len(modFiles) >= modLimit {
			return errDone
		}
		searched++
		if fileLimit > 0 && searched >= fileLimit {
			return errExhausted
		}
		return nil
	})
	if err == errDone {
		return modFiles, nil
	}
	return modFiles, err
}
