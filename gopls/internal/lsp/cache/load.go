// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"sort"
	"strings"
	"sync/atomic"
	"time"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/gopls/internal/bug"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/event/tag"
	"golang.org/x/tools/internal/gocommand"
	"golang.org/x/tools/internal/packagesinternal"
)

var loadID uint64 // atomic identifier for loads

// errNoPackages indicates that a load query matched no packages.
var errNoPackages = errors.New("no packages returned")

// load calls packages.Load for the given scopes, updating package metadata,
// import graph, and mapped files with the result.
//
// The resulting error may wrap the moduleErrorMap error type, representing
// errors associated with specific modules.
//
// If scopes contains a file scope there must be exactly one scope.
func (s *snapshot) load(ctx context.Context, allowNetwork bool, scopes ...loadScope) (err error) {
	id := atomic.AddUint64(&loadID, 1)
	eventName := fmt.Sprintf("go/packages.Load #%d", id) // unique name for logging

	var query []string
	var containsDir bool // for logging
	var standalone bool  // whether this is a load of a standalone file

	// Keep track of module query -> module path so that we can later correlate query
	// errors with errors.
	moduleQueries := make(map[string]string)
	for _, scope := range scopes {
		switch scope := scope.(type) {
		case packageLoadScope:
			// The only time we pass package paths is when we're doing a
			// partial workspace load. In those cases, the paths came back from
			// go list and should already be GOPATH-vendorized when appropriate.
			query = append(query, string(scope))

		case fileLoadScope:
			// Given multiple scopes, the resulting load might contain inaccurate
			// information. For example go/packages returns at most one command-line
			// arguments package, and does not handle a combination of standalone
			// files and packages.
			uri := span.URI(scope)
			if len(scopes) > 1 {
				panic(fmt.Sprintf("internal error: load called with multiple scopes when a file scope is present (file: %s)", uri))
			}
			fh := s.FindFile(uri)
			if fh == nil || s.FileKind(fh) != source.Go {
				// Don't try to load a file that doesn't exist, or isn't a go file.
				continue
			}
			contents, err := fh.Content()
			if err != nil {
				continue
			}
			if isStandaloneFile(contents, s.options.StandaloneTags) {
				standalone = true
				query = append(query, uri.Filename())
			} else {
				query = append(query, fmt.Sprintf("file=%s", uri.Filename()))
			}

		case moduleLoadScope:
			modQuery := fmt.Sprintf("%s%c...", scope.dir, filepath.Separator)
			query = append(query, modQuery)
			moduleQueries[modQuery] = string(scope.modulePath)

		case viewLoadScope:
			// If we are outside of GOPATH, a module, or some other known
			// build system, don't load subdirectories.
			if !s.validBuildConfiguration() {
				query = append(query, "./")
			} else {
				query = append(query, "./...")
			}

		default:
			panic(fmt.Sprintf("unknown scope type %T", scope))
		}
		switch scope.(type) {
		case viewLoadScope, moduleLoadScope:
			containsDir = true
		}
	}
	if len(query) == 0 {
		return nil
	}
	sort.Strings(query) // for determinism

	ctx, done := event.Start(ctx, "cache.snapshot.load", tag.Query.Of(query))
	defer done()

	flags := source.LoadWorkspace
	if allowNetwork {
		flags |= source.AllowNetwork
	}
	_, inv, cleanup, err := s.goCommandInvocation(ctx, flags, &gocommand.Invocation{
		WorkingDir: s.view.goCommandDir.Filename(),
	})
	if err != nil {
		return err
	}

	// Set a last resort deadline on packages.Load since it calls the go
	// command, which may hang indefinitely if it has a bug. golang/go#42132
	// and golang/go#42255 have more context.
	ctx, cancel := context.WithTimeout(ctx, 10*time.Minute)
	defer cancel()

	cfg := s.config(ctx, inv)
	pkgs, err := packages.Load(cfg, query...)
	cleanup()

	// If the context was canceled, return early. Otherwise, we might be
	// type-checking an incomplete result. Check the context directly,
	// because go/packages adds extra information to the error.
	if ctx.Err() != nil {
		return ctx.Err()
	}

	// This log message is sought for by TestReloadOnlyOnce.
	labels := append(source.SnapshotLabels(s), tag.Query.Of(query), tag.PackageCount.Of(len(pkgs)))
	if err != nil {
		event.Error(ctx, eventName, err, labels...)
	} else {
		event.Log(ctx, eventName, labels...)
	}

	if len(pkgs) == 0 {
		if err == nil {
			err = errNoPackages
		}
		return fmt.Errorf("packages.Load error: %w", err)
	}

	if standalone && len(pkgs) > 1 {
		return bug.Errorf("internal error: go/packages returned multiple packages for standalone file")
	}

	moduleErrs := make(map[string][]packages.Error) // module path -> errors
	filterFunc := s.filterFunc()
	newMetadata := make(map[PackageID]*source.Metadata)
	for _, pkg := range pkgs {
		// The Go command returns synthetic list results for module queries that
		// encountered module errors.
		//
		// For example, given a module path a.mod, we'll query for "a.mod/..." and
		// the go command will return a package named "a.mod/..." holding this
		// error. Save it for later interpretation.
		//
		// See golang/go#50862 for more details.
		if mod := moduleQueries[pkg.PkgPath]; mod != "" { // a synthetic result for the unloadable module
			if len(pkg.Errors) > 0 {
				moduleErrs[mod] = pkg.Errors
			}
			continue
		}

		if !containsDir || s.options.VerboseOutput {
			event.Log(ctx, eventName, append(
				source.SnapshotLabels(s),
				tag.Package.Of(pkg.ID),
				tag.Files.Of(pkg.CompiledGoFiles))...)
		}

		// Ignore packages with no sources, since we will never be able to
		// correctly invalidate that metadata.
		if len(pkg.GoFiles) == 0 && len(pkg.CompiledGoFiles) == 0 {
			continue
		}
		// Special case for the builtin package, as it has no dependencies.
		if pkg.PkgPath == "builtin" {
			if len(pkg.GoFiles) != 1 {
				return fmt.Errorf("only expected 1 file for builtin, got %v", len(pkg.GoFiles))
			}
			s.setBuiltin(pkg.GoFiles[0])
			continue
		}
		// Skip test main packages.
		if isTestMain(pkg, s.view.gocache) {
			continue
		}
		// Skip filtered packages. They may be added anyway if they're
		// dependencies of non-filtered packages.
		//
		// TODO(rfindley): why exclude metadata arbitrarily here? It should be safe
		// to capture all metadata.
		// TODO(rfindley): what about compiled go files?
		if allFilesExcluded(pkg.GoFiles, filterFunc) {
			continue
		}
		buildMetadata(newMetadata, pkg, cfg.Dir, standalone)
	}

	s.mu.Lock()

	// Assert the invariant s.packages.Get(id).m == s.meta.metadata[id].
	s.packages.Range(func(id PackageID, ph *packageHandle) {
		if s.meta.metadata[id] != ph.m {
			panic("inconsistent metadata")
		}
	})

	// Compute the minimal metadata updates (for Clone)
	// required to preserve the above invariant.
	var files []span.URI // files to preload
	seenFiles := make(map[span.URI]bool)
	updates := make(map[PackageID]*source.Metadata)
	for _, m := range newMetadata {
		if existing := s.meta.metadata[m.ID]; existing == nil {
			// Record any new files we should pre-load.
			for _, uri := range m.CompiledGoFiles {
				if !seenFiles[uri] {
					seenFiles[uri] = true
					files = append(files, uri)
				}
			}
			updates[m.ID] = m
			delete(s.shouldLoad, m.ID)
		}
	}

	event.Log(ctx, fmt.Sprintf("%s: updating metadata for %d packages", eventName, len(updates)))

	// Before mutating the snapshot, ensure that we compute load diagnostics
	// successfully. This could fail if the context is cancelled, and we don't
	// want to leave the snapshot metadata in a partial state.
	meta := s.meta.Clone(updates)
	workspacePackages := computeWorkspacePackagesLocked(s, meta)
	for _, update := range updates {
		computeLoadDiagnostics(ctx, update, meta, lockedSnapshot{s}, workspacePackages)
	}
	s.meta = meta
	s.workspacePackages = workspacePackages
	s.resetActivePackagesLocked()

	s.mu.Unlock()

	// Opt: preLoad files in parallel.
	//
	// Requesting files in batch optimizes the underlying filesystem reads.
	// However, this is also currently necessary for correctness: populating all
	// files in the snapshot is necessary for certain operations that rely on the
	// completeness of the file map, e.g. computing the set of directories to
	// watch.
	//
	// TODO(rfindley, golang/go#57558): determine the set of directories based on
	// loaded packages, so that reading files here is not necessary for
	// correctness.
	s.preloadFiles(ctx, files)

	if len(moduleErrs) > 0 {
		return &moduleErrorMap{moduleErrs}
	}

	return nil
}

type moduleErrorMap struct {
	errs map[string][]packages.Error // module path -> errors
}

func (m *moduleErrorMap) Error() string {
	var paths []string // sort for stability
	for path, errs := range m.errs {
		if len(errs) > 0 { // should always be true, but be cautious
			paths = append(paths, path)
		}
	}
	sort.Strings(paths)

	var buf bytes.Buffer
	fmt.Fprintf(&buf, "%d modules have errors:\n", len(paths))
	for _, path := range paths {
		fmt.Fprintf(&buf, "\t%s:%s\n", path, m.errs[path][0].Msg)
	}

	return buf.String()
}

// workspaceLayoutError returns an error describing a misconfiguration of the
// workspace, along with related diagnostic.
//
// The unusual argument ordering of results is intentional: if the resulting
// error is nil, so must be the resulting diagnostics.
//
// If ctx is cancelled, it may return ctx.Err(), nil.
//
// TODO(rfindley): separate workspace diagnostics from critical workspace
// errors.
func (s *snapshot) workspaceLayoutError(ctx context.Context) (error, []*source.Diagnostic) {
	// TODO(rfindley): both of the checks below should be delegated to the workspace.

	if s.view.effectiveGO111MODULE() == off {
		return nil, nil
	}

	// If the user is using a go.work file, we assume that they know what they
	// are doing.
	//
	// TODO(golang/go#53880): improve orphaned file diagnostics when using go.work.
	if s.view.gowork != "" {
		return nil, nil
	}

	// Apply diagnostics about the workspace configuration to relevant open
	// files.
	openFiles := s.overlays()

	// If the snapshot does not have a valid build configuration, it may be
	// that the user has opened a directory that contains multiple modules.
	// Check for that an warn about it.
	if !s.validBuildConfiguration() {
		var msg string
		if s.view.goversion >= 18 {
			msg = `gopls was not able to find modules in your workspace.
When outside of GOPATH, gopls needs to know which modules you are working on.
You can fix this by opening your workspace to a folder inside a Go module, or
by using a go.work file to specify multiple modules.
See the documentation for more information on setting up your workspace:
https://github.com/golang/tools/blob/master/gopls/doc/workspace.md.`
		} else {
			msg = `gopls requires a module at the root of your workspace.
You can work with multiple modules by upgrading to Go 1.18 or later, and using
go workspaces (go.work files).
See the documentation for more information on setting up your workspace:
https://github.com/golang/tools/blob/master/gopls/doc/workspace.md.`
		}
		return fmt.Errorf(msg), s.applyCriticalErrorToFiles(ctx, msg, openFiles)
	}

	return nil, nil
}

func (s *snapshot) applyCriticalErrorToFiles(ctx context.Context, msg string, files []*Overlay) []*source.Diagnostic {
	var srcDiags []*source.Diagnostic
	for _, fh := range files {
		// Place the diagnostics on the package or module declarations.
		var rng protocol.Range
		switch s.FileKind(fh) {
		case source.Go:
			if pgf, err := s.ParseGo(ctx, fh, source.ParseHeader); err == nil {
				// Check that we have a valid `package foo` range to use for positioning the error.
				if pgf.File.Package.IsValid() && pgf.File.Name != nil && pgf.File.Name.End().IsValid() {
					rng, _ = pgf.PosRange(pgf.File.Package, pgf.File.Name.End())
				}
			}
		case source.Mod:
			if pmf, err := s.ParseMod(ctx, fh); err == nil {
				if mod := pmf.File.Module; mod != nil && mod.Syntax != nil {
					rng, _ = pmf.Mapper.OffsetRange(mod.Syntax.Start.Byte, mod.Syntax.End.Byte)
				}
			}
		}
		srcDiags = append(srcDiags, &source.Diagnostic{
			URI:      fh.URI(),
			Range:    rng,
			Severity: protocol.SeverityError,
			Source:   source.ListError,
			Message:  msg,
		})
	}
	return srcDiags
}

// buildMetadata populates the updates map with metadata updates to
// apply, based on the given pkg. It recurs through pkg.Imports to ensure that
// metadata exists for all dependencies.
func buildMetadata(updates map[PackageID]*source.Metadata, pkg *packages.Package, loadDir string, standalone bool) {
	// Allow for multiple ad-hoc packages in the workspace (see #47584).
	pkgPath := PackagePath(pkg.PkgPath)
	id := PackageID(pkg.ID)

	if source.IsCommandLineArguments(id) {
		if len(pkg.CompiledGoFiles) != 1 {
			bug.Reportf("unexpected files in command-line-arguments package: %v", pkg.CompiledGoFiles)
			return
		}
		suffix := pkg.CompiledGoFiles[0]
		id = PackageID(pkg.ID + suffix)
		pkgPath = PackagePath(pkg.PkgPath + suffix)
	}

	// Duplicate?
	if _, ok := updates[id]; ok {
		// A package was encountered twice due to shared
		// subgraphs (common) or cycles (rare). Although "go
		// list" usually breaks cycles, we don't rely on it.
		// breakImportCycles in metadataGraph.Clone takes care
		// of it later.
		return
	}

	// Recreate the metadata rather than reusing it to avoid locking.
	m := &source.Metadata{
		ID:         id,
		PkgPath:    pkgPath,
		Name:       PackageName(pkg.Name),
		ForTest:    PackagePath(packagesinternal.GetForTest(pkg)),
		TypesSizes: pkg.TypesSizes,
		LoadDir:    loadDir,
		Module:     pkg.Module,
		Errors:     pkg.Errors,
		DepsErrors: packagesinternal.GetDepsErrors(pkg),
		Standalone: standalone,
	}

	updates[id] = m

	for _, filename := range pkg.CompiledGoFiles {
		uri := span.URIFromPath(filename)
		m.CompiledGoFiles = append(m.CompiledGoFiles, uri)
	}
	for _, filename := range pkg.GoFiles {
		uri := span.URIFromPath(filename)
		m.GoFiles = append(m.GoFiles, uri)
	}
	for _, filename := range pkg.IgnoredFiles {
		uri := span.URIFromPath(filename)
		m.IgnoredFiles = append(m.IgnoredFiles, uri)
	}

	depsByImpPath := make(map[ImportPath]PackageID)
	depsByPkgPath := make(map[PackagePath]PackageID)
	for importPath, imported := range pkg.Imports {
		importPath := ImportPath(importPath)

		// It is not an invariant that importPath == imported.PkgPath.
		// For example, package "net" imports "golang.org/x/net/dns/dnsmessage"
		// which refers to the package whose ID and PkgPath are both
		// "vendor/golang.org/x/net/dns/dnsmessage". Notice the ImportMap,
		// which maps ImportPaths to PackagePaths:
		//
		// $ go list -json net vendor/golang.org/x/net/dns/dnsmessage
		// {
		// 	"ImportPath": "net",
		// 	"Name": "net",
		// 	"Imports": [
		// 		"C",
		// 		"vendor/golang.org/x/net/dns/dnsmessage",
		// 		"vendor/golang.org/x/net/route",
		// 		...
		// 	],
		// 	"ImportMap": {
		// 		"golang.org/x/net/dns/dnsmessage": "vendor/golang.org/x/net/dns/dnsmessage",
		// 		"golang.org/x/net/route": "vendor/golang.org/x/net/route"
		// 	},
		//      ...
		// }
		// {
		// 	"ImportPath": "vendor/golang.org/x/net/dns/dnsmessage",
		// 	"Name": "dnsmessage",
		//      ...
		// }
		//
		// (Beware that, for historical reasons, go list uses
		// the JSON field "ImportPath" for the package's
		// path--effectively the linker symbol prefix.)
		//
		// The example above is slightly special to go list
		// because it's in the std module.  Otherwise,
		// vendored modules are simply modules whose directory
		// is vendor/ instead of GOMODCACHE, and the
		// import path equals the package path.
		//
		// But in GOPATH (non-module) mode, it's possible for
		// package vendoring to cause a non-identity ImportMap,
		// as in this example:
		//
		// $ cd $HOME/src
		// $ find . -type f
		// ./b/b.go
		// ./vendor/example.com/a/a.go
		// $ cat ./b/b.go
		// package b
		// import _ "example.com/a"
		// $ cat ./vendor/example.com/a/a.go
		// package a
		// $ GOPATH=$HOME GO111MODULE=off go list -json ./b | grep -A2 ImportMap
		//     "ImportMap": {
		//         "example.com/a": "vendor/example.com/a"
		//     },

		// Don't remember any imports with significant errors.
		//
		// The len=0 condition is a heuristic check for imports of
		// non-existent packages (for which go/packages will create
		// an edge to a synthesized node). The heuristic is unsound
		// because some valid packages have zero files, for example,
		// a directory containing only the file p_test.go defines an
		// empty package p.
		// TODO(adonovan): clarify this. Perhaps go/packages should
		// report which nodes were synthesized.
		if importPath != "unsafe" && len(imported.CompiledGoFiles) == 0 {
			depsByImpPath[importPath] = "" // missing
			continue
		}

		// Don't record self-import edges.
		// (This simplifies metadataGraph's cycle check.)
		if PackageID(imported.ID) == id {
			if len(pkg.Errors) == 0 {
				bug.Reportf("self-import without error in package %s", id)
			}
			continue
		}

		buildMetadata(updates, imported, loadDir, false) // only top level packages can be standalone

		// Don't record edges to packages with no name, as they cause trouble for
		// the importer (golang/go#60952).
		//
		// However, we do want to insert these packages into the update map
		// (buildMetadata above), so that we get type-checking diagnostics for the
		// invalid packages.
		if imported.Name == "" {
			depsByImpPath[importPath] = "" // missing
			continue
		}

		depsByImpPath[importPath] = PackageID(imported.ID)
		depsByPkgPath[PackagePath(imported.PkgPath)] = PackageID(imported.ID)
	}
	m.DepsByImpPath = depsByImpPath
	m.DepsByPkgPath = depsByPkgPath

	// m.Diagnostics is set later in the loading pass, using
	// computeLoadDiagnostics.
}

// computeLoadDiagnostics computes and sets m.Diagnostics for the given metadata m.
//
// It should only be called during metadata construction in snapshot.load.
func computeLoadDiagnostics(ctx context.Context, m *source.Metadata, meta *metadataGraph, fs source.FileSource, workspacePackages map[PackageID]PackagePath) {
	for _, packagesErr := range m.Errors {
		// Filter out parse errors from go list. We'll get them when we
		// actually parse, and buggy overlay support may generate spurious
		// errors. (See TestNewModule_Issue38207.)
		if strings.Contains(packagesErr.Msg, "expected '") {
			continue
		}
		pkgDiags, err := goPackagesErrorDiagnostics(ctx, packagesErr, m, fs)
		if err != nil {
			// There are certain cases where the go command returns invalid
			// positions, so we cannot panic or even bug.Reportf here.
			event.Error(ctx, "unable to compute positions for list errors", err, tag.Package.Of(string(m.ID)))
			continue
		}
		m.Diagnostics = append(m.Diagnostics, pkgDiags...)
	}

	// TODO(rfindley): this is buggy: an insignificant change to a modfile
	// (or an unsaved modfile) could affect the position of deps errors,
	// without invalidating the package.
	depsDiags, err := depsErrors(ctx, m, meta, fs, workspacePackages)
	if err != nil {
		if ctx.Err() == nil {
			// TODO(rfindley): consider making this a bug.Reportf. depsErrors should
			// not normally fail.
			event.Error(ctx, "unable to compute deps errors", err, tag.Package.Of(string(m.ID)))
		}
	}
	m.Diagnostics = append(m.Diagnostics, depsDiags...)
}

// containsPackageLocked reports whether p is a workspace package for the
// snapshot s.
//
// s.mu must be held while calling this function.
func containsPackageLocked(s *snapshot, m *source.Metadata) bool {
	// In legacy workspace mode, or if a package does not have an associated
	// module, a package is considered inside the workspace if any of its files
	// are under the workspace root (and not excluded).
	//
	// Otherwise if the package has a module it must be an active module (as
	// defined by the module root or go.work file) and at least one file must not
	// be filtered out by directoryFilters.
	//
	// TODO(rfindley): revisit this function. We should not need to predicate on
	// gowork != "". It should suffice to consider workspace mod files (also, we
	// will hopefully eliminate the concept of a workspace package soon).
	if m.Module != nil && s.view.gowork != "" {
		modURI := span.URIFromPath(m.Module.GoMod)
		_, ok := s.workspaceModFiles[modURI]
		if !ok {
			return false
		}

		uris := map[span.URI]struct{}{}
		for _, uri := range m.CompiledGoFiles {
			uris[uri] = struct{}{}
		}
		for _, uri := range m.GoFiles {
			uris[uri] = struct{}{}
		}

		filterFunc := s.filterFunc()
		for uri := range uris {
			// Don't use view.contains here. go.work files may include modules
			// outside of the workspace folder.
			if !strings.Contains(string(uri), "/vendor/") && !filterFunc(uri) {
				return true
			}
		}
		return false
	}

	return containsFileInWorkspaceLocked(s, m)
}

// containsOpenFileLocked reports whether any file referenced by m is open in
// the snapshot s.
//
// s.mu must be held while calling this function.
func containsOpenFileLocked(s *snapshot, m *source.Metadata) bool {
	uris := map[span.URI]struct{}{}
	for _, uri := range m.CompiledGoFiles {
		uris[uri] = struct{}{}
	}
	for _, uri := range m.GoFiles {
		uris[uri] = struct{}{}
	}

	for uri := range uris {
		fh, _ := s.files.Get(uri)
		if _, open := fh.(*Overlay); open {
			return true
		}
	}
	return false
}

// containsFileInWorkspaceLocked reports whether m contains any file inside the
// workspace of the snapshot s.
//
// s.mu must be held while calling this function.
func containsFileInWorkspaceLocked(s *snapshot, m *source.Metadata) bool {
	uris := map[span.URI]struct{}{}
	for _, uri := range m.CompiledGoFiles {
		uris[uri] = struct{}{}
	}
	for _, uri := range m.GoFiles {
		uris[uri] = struct{}{}
	}

	for uri := range uris {
		// In order for a package to be considered for the workspace, at least one
		// file must be contained in the workspace and not vendored.

		// The package's files are in this view. It may be a workspace package.
		// Vendored packages are not likely to be interesting to the user.
		if !strings.Contains(string(uri), "/vendor/") && s.contains(uri) {
			return true
		}
	}
	return false
}

// computeWorkspacePackagesLocked computes workspace packages in the
// snapshot s for the given metadata graph. The result does not
// contain intermediate test variants.
//
// s.mu must be held while calling this function.
func computeWorkspacePackagesLocked(s *snapshot, meta *metadataGraph) map[PackageID]PackagePath {
	workspacePackages := make(map[PackageID]PackagePath)
	for _, m := range meta.metadata {
		if !containsPackageLocked(s, m) {
			continue
		}

		if source.IsCommandLineArguments(m.ID) {
			// If all the files contained in m have a real package, we don't need to
			// keep m as a workspace package.
			if allFilesHaveRealPackages(meta, m) {
				continue
			}

			// We only care about command-line-arguments packages if they are still
			// open.
			if !containsOpenFileLocked(s, m) {
				continue
			}
		}

		switch {
		case m.ForTest == "":
			// A normal package.
			workspacePackages[m.ID] = m.PkgPath
		case m.ForTest == m.PkgPath, m.ForTest+"_test" == m.PkgPath:
			// The test variant of some workspace package or its x_test.
			// To load it, we need to load the non-test variant with -test.
			//
			// Notably, this excludes intermediate test variants from workspace
			// packages.
			assert(!m.IsIntermediateTestVariant(), "unexpected ITV")
			workspacePackages[m.ID] = m.ForTest
		}
	}
	return workspacePackages
}

// allFilesHaveRealPackages reports whether all files referenced by m are
// contained in a "real" package (not command-line-arguments).
//
// If m is valid but all "real" packages containing any file are invalid, this
// function returns false.
//
// If m is not a command-line-arguments package, this is trivially true.
func allFilesHaveRealPackages(g *metadataGraph, m *source.Metadata) bool {
	n := len(m.CompiledGoFiles)
checkURIs:
	for _, uri := range append(m.CompiledGoFiles[0:n:n], m.GoFiles...) {
		for _, id := range g.ids[uri] {
			if !source.IsCommandLineArguments(id) {
				continue checkURIs
			}
		}
		return false
	}
	return true
}

func isTestMain(pkg *packages.Package, gocache string) bool {
	// Test mains must have an import path that ends with ".test".
	if !strings.HasSuffix(pkg.PkgPath, ".test") {
		return false
	}
	// Test main packages are always named "main".
	if pkg.Name != "main" {
		return false
	}
	// Test mains always have exactly one GoFile that is in the build cache.
	if len(pkg.GoFiles) > 1 {
		return false
	}
	if !source.InDir(gocache, pkg.GoFiles[0]) {
		return false
	}
	return true
}
