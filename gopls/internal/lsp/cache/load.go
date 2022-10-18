// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync/atomic"
	"time"

	"golang.org/x/tools/go/packages"
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
func (s *snapshot) load(ctx context.Context, allowNetwork bool, scopes ...loadScope) (err error) {
	id := atomic.AddUint64(&loadID, 1)
	eventName := fmt.Sprintf("go/packages.Load #%d", id) // unique name for logging

	var query []string
	var containsDir bool // for logging

	// Keep track of module query -> module path so that we can later correlate query
	// errors with errors.
	moduleQueries := make(map[string]string)
	for _, scope := range scopes {
		switch scope := scope.(type) {
		case packageLoadScope:
			if source.IsCommandLineArguments(string(scope)) {
				panic("attempted to load command-line-arguments")
			}
			// The only time we pass package paths is when we're doing a
			// partial workspace load. In those cases, the paths came back from
			// go list and should already be GOPATH-vendorized when appropriate.
			query = append(query, string(scope))

		case fileLoadScope:
			uri := span.URI(scope)
			fh := s.FindFile(uri)
			if fh == nil || s.View().FileKind(fh) != source.Go {
				// Don't try to load a file that doesn't exist, or isn't a go file.
				continue
			}
			contents, err := fh.Read()
			if err != nil {
				continue
			}
			if isStandaloneFile(contents, s.view.Options().StandaloneTags) {
				query = append(query, uri.Filename())
			} else {
				query = append(query, fmt.Sprintf("file=%s", uri.Filename()))
			}

		case moduleLoadScope:
			switch scope {
			case "std", "cmd":
				query = append(query, string(scope))
			default:
				modQuery := fmt.Sprintf("%s/...", scope)
				query = append(query, modQuery)
				moduleQueries[modQuery] = string(scope)
			}

		case viewLoadScope:
			// If we are outside of GOPATH, a module, or some other known
			// build system, don't load subdirectories.
			if !s.ValidBuildConfiguration() {
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

	if s.view.Options().VerboseWorkDoneProgress {
		work := s.view.session.progress.Start(ctx, "Load", fmt.Sprintf("Loading query=%s", query), nil, nil)
		defer work.End(ctx, "Done.")
	}

	ctx, done := event.Start(ctx, "cache.view.load", tag.Query.Of(query))
	defer done()

	flags := source.LoadWorkspace
	if allowNetwork {
		flags |= source.AllowNetwork
	}
	_, inv, cleanup, err := s.goCommandInvocation(ctx, flags, &gocommand.Invocation{
		WorkingDir: s.view.rootURI.Filename(),
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
	if err != nil {
		event.Error(ctx, eventName, err, tag.Snapshot.Of(s.ID()), tag.Directory.Of(cfg.Dir), tag.Query.Of(query), tag.PackageCount.Of(len(pkgs)))
	} else {
		event.Log(ctx, eventName, tag.Snapshot.Of(s.ID()), tag.Directory.Of(cfg.Dir), tag.Query.Of(query), tag.PackageCount.Of(len(pkgs)))
	}

	if len(pkgs) == 0 {
		if err == nil {
			err = errNoPackages
		}
		return fmt.Errorf("packages.Load error: %w", err)
	}

	moduleErrs := make(map[string][]packages.Error) // module path -> errors
	filterer := buildFilterer(s.view.rootURI.Filename(), s.view.gomodcache, s.view.Options())
	newMetadata := make(map[PackageID]*KnownMetadata)
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

		if !containsDir || s.view.Options().VerboseOutput {
			event.Log(ctx, eventName,
				tag.Snapshot.Of(s.ID()),
				tag.Package.Of(pkg.ID),
				tag.Files.Of(pkg.CompiledGoFiles))
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
		if s.view.allFilesExcluded(pkg, filterer) {
			continue
		}
		if err := buildMetadata(ctx, pkg, cfg, query, newMetadata, nil); err != nil {
			return err
		}
	}

	s.mu.Lock()

	// Only update metadata where we don't already have valid metadata.
	//
	// We want to preserve an invariant that s.packages.Get(id).m.Metadata
	// matches s.meta.metadata[id].Metadata. By avoiding overwriting valid
	// metadata, we minimize the amount of invalidation required to preserve this
	// invariant.
	//
	// TODO(rfindley): perform a sanity check that metadata matches here. If not,
	// we have an invalidation bug elsewhere.
	updates := make(map[PackageID]*KnownMetadata)
	var updatedIDs []PackageID
	for _, m := range newMetadata {
		if existing := s.meta.metadata[m.ID]; existing == nil || !existing.Valid {
			updates[m.ID] = m
			updatedIDs = append(updatedIDs, m.ID)
			delete(s.shouldLoad, m.ID)
		}
	}

	event.Log(ctx, fmt.Sprintf("%s: updating metadata for %d packages", eventName, len(updates)))

	// Invalidate the reverse transitive closure of packages that have changed.
	//
	// Note that the original metadata is being invalidated here, so we use the
	// original metadata graph to compute the reverse closure.
	invalidatedPackages := s.meta.reverseTransitiveClosure(true, updatedIDs...)

	s.meta = s.meta.Clone(updates)
	s.resetIsActivePackageLocked()

	// Invalidate any packages and analysis results we may have associated with
	// this metadata.
	//
	// Generally speaking we should have already invalidated these results in
	// snapshot.clone, but with experimentalUseInvalidMetadata is may be possible
	// that we have re-computed stale results before the reload completes. In
	// this case, we must re-invalidate here.
	//
	// TODO(golang/go#54180): if we decide to make experimentalUseInvalidMetadata
	// obsolete, we should avoid this invalidation.
	s.invalidatePackagesLocked(invalidatedPackages)

	s.workspacePackages = computeWorkspacePackagesLocked(s, s.meta)
	s.dumpWorkspace("load")
	s.mu.Unlock()

	// Recompute the workspace package handle for any packages we invalidated.
	//
	// This is (putatively) an optimization since handle
	// construction prefetches the content of all Go source files.
	// It is safe to ignore errors, or omit this step entirely.
	for _, m := range updates {
		s.buildPackageHandle(ctx, m.ID, s.workspaceParseMode(m.ID)) // ignore error
	}

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

// workspaceLayoutErrors returns a diagnostic for every open file, as well as
// an error message if there are no open files.
func (s *snapshot) workspaceLayoutError(ctx context.Context) *source.CriticalError {
	// TODO(rfindley): do we really not want to show a critical error if the user
	// has no go.mod files?
	if len(s.workspace.getKnownModFiles()) == 0 {
		return nil
	}

	// TODO(rfindley): both of the checks below should be delegated to the workspace.
	if s.view.userGo111Module == off {
		return nil
	}
	if s.workspace.moduleSource != legacyWorkspace {
		return nil
	}

	// If the user has one module per view, there is nothing to warn about.
	if s.ValidBuildConfiguration() && len(s.workspace.getKnownModFiles()) == 1 {
		return nil
	}

	// Apply diagnostics about the workspace configuration to relevant open
	// files.
	openFiles := s.openFiles()

	// If the snapshot does not have a valid build configuration, it may be
	// that the user has opened a directory that contains multiple modules.
	// Check for that an warn about it.
	if !s.ValidBuildConfiguration() {
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
		return &source.CriticalError{
			MainError:   fmt.Errorf(msg),
			Diagnostics: s.applyCriticalErrorToFiles(ctx, msg, openFiles),
		}
	}

	// If the user has one active go.mod file, they may still be editing files
	// in nested modules. Check the module of each open file and add warnings
	// that the nested module must be opened as a workspace folder.
	if len(s.workspace.getActiveModFiles()) == 1 {
		// Get the active root go.mod file to compare against.
		var rootModURI span.URI
		for uri := range s.workspace.getActiveModFiles() {
			rootModURI = uri
		}
		nestedModules := map[string][]source.VersionedFileHandle{}
		for _, fh := range openFiles {
			modURI := moduleForURI(s.workspace.knownModFiles, fh.URI())
			if modURI != rootModURI {
				modDir := filepath.Dir(modURI.Filename())
				nestedModules[modDir] = append(nestedModules[modDir], fh)
			}
		}
		// Add a diagnostic to each file in a nested module to mark it as
		// "orphaned". Don't show a general diagnostic in the progress bar,
		// because the user may still want to edit a file in a nested module.
		var srcDiags []*source.Diagnostic
		for modDir, uris := range nestedModules {
			msg := fmt.Sprintf(`This file is in %s, which is a nested module in the %s module.
gopls currently requires one module per workspace folder.
Please open %s as a separate workspace folder.
You can learn more here: https://github.com/golang/tools/blob/master/gopls/doc/workspace.md.
`, modDir, filepath.Dir(rootModURI.Filename()), modDir)
			srcDiags = append(srcDiags, s.applyCriticalErrorToFiles(ctx, msg, uris)...)
		}
		if len(srcDiags) != 0 {
			return &source.CriticalError{
				MainError: fmt.Errorf(`You are working in a nested module.
Please open it as a separate workspace folder. Learn more:
https://github.com/golang/tools/blob/master/gopls/doc/workspace.md.`),
				Diagnostics: srcDiags,
			}
		}
	}
	return nil
}

func (s *snapshot) applyCriticalErrorToFiles(ctx context.Context, msg string, files []source.VersionedFileHandle) []*source.Diagnostic {
	var srcDiags []*source.Diagnostic
	for _, fh := range files {
		// Place the diagnostics on the package or module declarations.
		var rng protocol.Range
		switch s.view.FileKind(fh) {
		case source.Go:
			if pgf, err := s.ParseGo(ctx, fh, source.ParseHeader); err == nil {
				// Check that we have a valid `package foo` range to use for positioning the error.
				if pgf.File.Package.IsValid() && pgf.File.Name != nil && pgf.File.Name.End().IsValid() {
					pkgDecl := span.NewRange(pgf.Tok, pgf.File.Package, pgf.File.Name.End())
					if spn, err := pkgDecl.Span(); err == nil {
						rng, _ = pgf.Mapper.Range(spn)
					}
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

// getWorkspaceDir returns the URI for the workspace directory
// associated with this snapshot. The workspace directory is a
// temporary directory containing the go.mod file computed from all
// active modules.
func (s *snapshot) getWorkspaceDir(ctx context.Context) (span.URI, error) {
	s.mu.Lock()
	dir, err := s.workspaceDir, s.workspaceDirErr
	s.mu.Unlock()
	if dir == "" && err == nil { // cache miss
		dir, err = makeWorkspaceDir(ctx, s.workspace, s)
		s.mu.Lock()
		s.workspaceDir, s.workspaceDirErr = dir, err
		s.mu.Unlock()
	}
	return span.URIFromPath(dir), err
}

// makeWorkspaceDir creates a temporary directory containing a go.mod
// and go.sum file for each module in the workspace.
// Note: snapshot's mutex must be unlocked for it to satisfy FileSource.
func makeWorkspaceDir(ctx context.Context, workspace *workspace, fs source.FileSource) (string, error) {
	file, err := workspace.modFile(ctx, fs)
	if err != nil {
		return "", err
	}
	modContent, err := file.Format()
	if err != nil {
		return "", err
	}
	sumContent, err := workspace.sumFile(ctx, fs)
	if err != nil {
		return "", err
	}
	tmpdir, err := ioutil.TempDir("", "gopls-workspace-mod")
	if err != nil {
		return "", err
	}
	for name, content := range map[string][]byte{
		"go.mod": modContent,
		"go.sum": sumContent,
	} {
		if err := ioutil.WriteFile(filepath.Join(tmpdir, name), content, 0644); err != nil {
			os.RemoveAll(tmpdir) // ignore error
			return "", err
		}
	}
	return tmpdir, nil
}

// buildMetadata populates the updates map with metadata updates to
// apply, based on the given pkg. It recurs through pkg.Imports to ensure that
// metadata exists for all dependencies.
func buildMetadata(ctx context.Context, pkg *packages.Package, cfg *packages.Config, query []string, updates map[PackageID]*KnownMetadata, path []PackageID) error {
	// Allow for multiple ad-hoc packages in the workspace (see #47584).
	pkgPath := PackagePath(pkg.PkgPath)
	id := PackageID(pkg.ID)
	if source.IsCommandLineArguments(pkg.ID) {
		suffix := ":" + strings.Join(query, ",")
		id = PackageID(string(id) + suffix)
		pkgPath = PackagePath(string(pkgPath) + suffix)
	}

	if _, ok := updates[id]; ok {
		// If we've already seen this dependency, there may be an import cycle, or
		// we may have reached the same package transitively via distinct paths.
		// Check the path to confirm.

		// TODO(rfindley): this doesn't look sufficient. Any single piece of new
		// metadata could theoretically introduce import cycles in the metadata
		// graph. What's the point of this limited check here (and is it even
		// possible to get an import cycle in data from go/packages)? Consider
		// simply returning, so that this function need not return an error.
		//
		// We should consider doing a more complete guard against import cycles
		// elsewhere.
		for _, prev := range path {
			if prev == id {
				return fmt.Errorf("import cycle detected: %q", id)
			}
		}
		return nil
	}

	// Recreate the metadata rather than reusing it to avoid locking.
	m := &KnownMetadata{
		Metadata: &Metadata{
			ID:         id,
			PkgPath:    pkgPath,
			Name:       PackageName(pkg.Name),
			ForTest:    PackagePath(packagesinternal.GetForTest(pkg)),
			TypesSizes: pkg.TypesSizes,
			Config:     cfg,
			Module:     pkg.Module,
			depsErrors: packagesinternal.GetDepsErrors(pkg),
		},
		Valid: true,
	}
	updates[id] = m

	for _, err := range pkg.Errors {
		// Filter out parse errors from go list. We'll get them when we
		// actually parse, and buggy overlay support may generate spurious
		// errors. (See TestNewModule_Issue38207.)
		if strings.Contains(err.Msg, "expected '") {
			continue
		}
		m.Errors = append(m.Errors, err)
	}

	for _, filename := range pkg.CompiledGoFiles {
		uri := span.URIFromPath(filename)
		m.CompiledGoFiles = append(m.CompiledGoFiles, uri)
	}
	for _, filename := range pkg.GoFiles {
		uri := span.URIFromPath(filename)
		m.GoFiles = append(m.GoFiles, uri)
	}

	imports := make(map[ImportPath]PackageID)
	for importPath, imported := range pkg.Imports {
		importPath := ImportPath(importPath)
		imports[importPath] = PackageID(imported.ID)

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
			if m.MissingDeps == nil {
				m.MissingDeps = make(map[ImportPath]struct{})
			}
			m.MissingDeps[importPath] = struct{}{}
			continue
		}

		if err := buildMetadata(ctx, imported, cfg, query, updates, append(path, id)); err != nil {
			event.Error(ctx, "error in dependency", err)
		}
	}
	m.Imports = imports

	return nil
}

// containsPackageLocked reports whether p is a workspace package for the
// snapshot s.
//
// s.mu must be held while calling this function.
func containsPackageLocked(s *snapshot, m *Metadata) bool {
	// In legacy workspace mode, or if a package does not have an associated
	// module, a package is considered inside the workspace if any of its files
	// are under the workspace root (and not excluded).
	//
	// Otherwise if the package has a module it must be an active module (as
	// defined by the module root or go.work file) and at least one file must not
	// be filtered out by directoryFilters.
	if m.Module != nil && s.workspace.moduleSource != legacyWorkspace {
		modURI := span.URIFromPath(m.Module.GoMod)
		_, ok := s.workspace.activeModFiles[modURI]
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

		filterFunc := s.view.filterFunc()
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
func containsOpenFileLocked(s *snapshot, m *KnownMetadata) bool {
	uris := map[span.URI]struct{}{}
	for _, uri := range m.CompiledGoFiles {
		uris[uri] = struct{}{}
	}
	for _, uri := range m.GoFiles {
		uris[uri] = struct{}{}
	}

	for uri := range uris {
		if s.isOpenLocked(uri) {
			return true
		}
	}
	return false
}

// containsFileInWorkspaceLocked reports whether m contains any file inside the
// workspace of the snapshot s.
//
// s.mu must be held while calling this function.
func containsFileInWorkspaceLocked(s *snapshot, m *Metadata) bool {
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
		if !strings.Contains(string(uri), "/vendor/") && s.view.contains(uri) {
			return true
		}
	}
	return false
}

// computeWorkspacePackagesLocked computes workspace packages in the snapshot s
// for the given metadata graph.
//
// s.mu must be held while calling this function.
func computeWorkspacePackagesLocked(s *snapshot, meta *metadataGraph) map[PackageID]PackagePath {
	workspacePackages := make(map[PackageID]PackagePath)
	for _, m := range meta.metadata {
		// Don't consider invalid packages to be workspace packages. Doing so can
		// result in type-checking and diagnosing packages that no longer exist,
		// which can lead to memory leaks and confusing errors.
		if !m.Valid {
			continue
		}

		if !containsPackageLocked(s, m.Metadata) {
			continue
		}

		if source.IsCommandLineArguments(string(m.ID)) {
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
func allFilesHaveRealPackages(g *metadataGraph, m *KnownMetadata) bool {
	n := len(m.CompiledGoFiles)
checkURIs:
	for _, uri := range append(m.CompiledGoFiles[0:n:n], m.GoFiles...) {
		for _, id := range g.ids[uri] {
			if !source.IsCommandLineArguments(string(id)) && (g.metadata[id].Valid || !m.Valid) {
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
