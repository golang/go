// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
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

// dirs returns the workspace directories for the loaded modules.
//
// A workspace directory is, roughly speaking, a directory for which we care
// about file changes. This is used for the purpose of registering file
// watching patterns, and expanding directory modifications to their adjacent
// files.
//
// TODO(rfindley): move this to snapshot.go.
// TODO(rfindley): can we make this abstraction simpler and/or more accurate?
func (s *snapshot) dirs(ctx context.Context) []span.URI {
	dirSet := make(map[span.URI]struct{})

	// Dirs should, at the very least, contain the working directory and folder.
	dirSet[s.view.workingDir()] = struct{}{}
	dirSet[s.view.folder] = struct{}{}

	// Additionally, if e.g. go.work indicates other workspace modules, we should
	// include their directories too.
	if s.workspaceModFilesErr == nil {
		for modFile := range s.workspaceModFiles {
			dir := filepath.Dir(modFile.Filename())
			dirSet[span.URIFromPath(dir)] = struct{}{}
		}
	}
	var dirs []span.URI
	for d := range dirSet {
		dirs = append(dirs, d)
	}
	sort.Slice(dirs, func(i, j int) bool { return dirs[i] < dirs[j] })
	return dirs
}

// isGoMod reports if uri is a go.mod file.
func isGoMod(uri span.URI) bool {
	return filepath.Base(uri.Filename()) == "go.mod"
}

// isGoWork reports if uri is a go.work file.
func isGoWork(uri span.URI) bool {
	return filepath.Base(uri.Filename()) == "go.work"
}

// fileExists reports if the file uri exists within source.
func fileExists(ctx context.Context, uri span.URI, source source.FileSource) (bool, error) {
	fh, err := source.ReadFile(ctx, uri)
	if err != nil {
		return false, err
	}
	return fileHandleExists(fh)
}

// fileHandleExists reports if the file underlying fh actually exists.
func fileHandleExists(fh source.FileHandle) (bool, error) {
	_, err := fh.Content()
	if err == nil {
		return true, nil
	}
	if os.IsNotExist(err) {
		return false, nil
	}
	return false, err
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
