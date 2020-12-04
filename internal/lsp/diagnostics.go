// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/mod"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/xcontext"
	errors "golang.org/x/xerrors"
)

// diagnosticSource differentiates different sources of diagnostics.
type diagnosticSource int

const (
	modSource diagnosticSource = iota
	gcDetailsSource
	analysisSource
	typeCheckSource
	orphanedSource
)

// A diagnosticReport holds results for a single diagnostic source.
type diagnosticReport struct {
	snapshotID    uint64
	publishedHash string
	diags         map[string]*source.Diagnostic
}

// fileReports holds a collection of diagnostic reports for a single file, as
// well as the hash of the last published set of diagnostics.
type fileReports struct {
	snapshotID    uint64
	publishedHash string
	reports       map[diagnosticSource]diagnosticReport
}

// hashDiagnostics computes a hash to identify diags.
func hashDiagnostics(diags ...*source.Diagnostic) string {
	source.SortDiagnostics(diags)
	h := sha256.New()
	for _, d := range diags {
		for _, t := range d.Tags {
			fmt.Fprintf(h, "%s", t)
		}
		for _, r := range d.Related {
			fmt.Fprintf(h, "%s%s%s", r.URI, r.Message, r.Range)
		}
		fmt.Fprintf(h, "%s%s%s%s", d.Message, d.Range, d.Severity, d.Source)
	}
	return fmt.Sprintf("%x", h.Sum(nil))
}

func (s *Server) diagnoseDetached(snapshot source.Snapshot) {
	ctx := snapshot.View().BackgroundContext()
	ctx = xcontext.Detach(ctx)
	showWarning := s.diagnose(ctx, snapshot, false)
	if showWarning {
		// If a view has been created or the configuration changed, warn the user.
		s.client.ShowMessage(ctx, &protocol.ShowMessageParams{
			Type:    protocol.Warning,
			Message: `You are neither in a module nor in your GOPATH. If you are using modules, please open your editor to a directory in your module. If you believe this warning is incorrect, please file an issue: https://github.com/golang/go/issues/new.`,
		})
	}
	s.publishDiagnostics(ctx, true, snapshot)
}

func (s *Server) diagnoseSnapshot(snapshot source.Snapshot, changedURIs []span.URI, onDisk bool) {
	ctx := snapshot.View().BackgroundContext()

	delay := snapshot.View().Options().ExperimentalDiagnosticsDelay
	if delay > 0 {
		// Experimental 2-phase diagnostics.
		//
		// The first phase just parses and checks packages that have been
		// affected by file modifications (no analysis).
		//
		// The second phase does everything, and is debounced by the configured
		// delay.
		s.diagnoseChangedFiles(ctx, snapshot, changedURIs, onDisk)
		s.publishDiagnostics(ctx, false, snapshot)
		s.debouncer.debounce(snapshot.View().Name(), snapshot.ID(), delay, func() {
			s.diagnose(ctx, snapshot, false)
			s.publishDiagnostics(ctx, true, snapshot)
		})
		return
	}

	// Ignore possible workspace configuration warnings in the normal flow.
	s.diagnose(ctx, snapshot, false)
	s.publishDiagnostics(ctx, true, snapshot)
}

func (s *Server) diagnoseChangedFiles(ctx context.Context, snapshot source.Snapshot, uris []span.URI, onDisk bool) {
	ctx, done := event.Start(ctx, "Server.diagnoseChangedFiles")
	defer done()
	packages := make(map[source.Package]struct{})
	for _, uri := range uris {
		// If the change is only on-disk and the file is not open, don't
		// directly request its package. It may not be a workspace package.
		if onDisk && !snapshot.IsOpen(uri) {
			continue
		}
		pkgs, err := snapshot.PackagesForFile(ctx, uri, source.TypecheckWorkspace)
		if err != nil {
			// TODO (findleyr): we should probably do something with the error here,
			// but as of now this can fail repeatedly if load fails, so can be too
			// noisy to log (and we'll handle things later in the slow pass).
			continue
		}
		for _, pkg := range pkgs {
			packages[pkg] = struct{}{}
		}
	}
	var wg sync.WaitGroup
	for pkg := range packages {
		wg.Add(1)

		go func(pkg source.Package) {
			defer wg.Done()

			_ = s.diagnosePkg(ctx, snapshot, pkg, true)
		}(pkg)
	}
	wg.Wait()
}

// diagnose is a helper function for running diagnostics with a given context.
// Do not call it directly.
func (s *Server) diagnose(ctx context.Context, snapshot source.Snapshot, alwaysAnalyze bool) bool {
	ctx, done := event.Start(ctx, "Server.diagnose")
	defer done()

	// Wait for a free diagnostics slot.
	select {
	case <-ctx.Done():
		return false
	case s.diagnosticsSema <- struct{}{}:
	}
	defer func() {
		<-s.diagnosticsSema
	}()

	// First, diagnose the go.mod file.
	modReports, modErr := mod.Diagnostics(ctx, snapshot)
	if ctx.Err() != nil {
		return false
	}
	if modErr != nil {
		event.Error(ctx, "warning: diagnose go.mod", modErr, tag.Directory.Of(snapshot.View().Folder().Filename()), tag.Snapshot.Of(snapshot.ID()))
	}
	for id, diags := range modReports {
		if id.URI == "" {
			event.Error(ctx, "missing URI for module diagnostics", fmt.Errorf("empty URI"), tag.Directory.Of(snapshot.View().Folder().Filename()))
			continue
		}
		s.storeDiagnostics(snapshot, id.URI, modSource, diags)
	}

	// Diagnose all of the packages in the workspace.
	wsPkgs, err := snapshot.WorkspacePackages(ctx)
	if s.shouldIgnoreError(ctx, snapshot, err) {
		return false
	}

	// Show the error as a progress error report so that it appears in the
	// status bar. If a client doesn't support progress reports, the error
	// will still be shown as a ShowMessage. If there is no error, any running
	// error progress reports will be closed.
	s.showCriticalErrorStatus(ctx, snapshot, err)

	if err != nil {
		event.Error(ctx, "errors diagnosing workspace", err, tag.Snapshot.Of(snapshot.ID()), tag.Directory.Of(snapshot.View().Folder()))
		return false
	}

	var (
		wg              sync.WaitGroup
		shouldShowMsgMu sync.Mutex
		shouldShowMsg   bool
		seen            = map[span.URI]struct{}{}
	)
	for _, pkg := range wsPkgs {
		wg.Add(1)

		for _, pgf := range pkg.CompiledGoFiles() {
			seen[pgf.URI] = struct{}{}
		}

		go func(pkg source.Package) {
			defer wg.Done()

			show := s.diagnosePkg(ctx, snapshot, pkg, alwaysAnalyze)
			if show {
				shouldShowMsgMu.Lock()
				shouldShowMsg = true
				shouldShowMsgMu.Unlock()
			}
		}(pkg)
	}
	wg.Wait()
	// Confirm that every opened file belongs to a package (if any exist in
	// the workspace). Otherwise, add a diagnostic to the file.
	if len(wsPkgs) > 0 {
		for _, o := range s.session.Overlays() {
			if _, ok := seen[o.URI()]; ok {
				continue
			}
			diagnostic := s.checkForOrphanedFile(ctx, snapshot, o)
			if diagnostic == nil {
				continue
			}
			s.storeDiagnostics(snapshot, o.URI(), orphanedSource, []*source.Diagnostic{diagnostic})
		}
	}
	return shouldShowMsg
}

func (s *Server) diagnosePkg(ctx context.Context, snapshot source.Snapshot, pkg source.Package, alwaysAnalyze bool) bool {
	includeAnalysis := alwaysAnalyze // only run analyses for packages with open files
	var gcDetailsDir span.URI        // find the package's optimization details, if available
	for _, pgf := range pkg.CompiledGoFiles() {
		if snapshot.IsOpen(pgf.URI) {
			includeAnalysis = true
		}
		if gcDetailsDir == "" {
			dirURI := span.URIFromPath(filepath.Dir(pgf.URI.Filename()))
			s.gcOptimizationDetailsMu.Lock()
			_, ok := s.gcOptimizationDetails[dirURI]
			s.gcOptimizationDetailsMu.Unlock()
			if ok {
				gcDetailsDir = dirURI
			}
		}
	}

	typeCheckResults := source.GetTypeCheckDiagnostics(ctx, snapshot, pkg)
	for uri, diags := range typeCheckResults.Diagnostics {
		s.storeDiagnostics(snapshot, uri, typeCheckSource, diags)
	}
	if includeAnalysis && !typeCheckResults.HasParseOrListErrors {
		reports, err := source.Analyze(ctx, snapshot, pkg, typeCheckResults)
		if err != nil {
			event.Error(ctx, "warning: diagnose package", err, tag.Snapshot.Of(snapshot.ID()), tag.Package.Of(pkg.ID()))
			return false
		}
		for uri, diags := range reports {
			s.storeDiagnostics(snapshot, uri, analysisSource, diags)
		}
	}
	// If gc optimization details are available, add them to the
	// diagnostic reports.
	if gcDetailsDir != "" {
		gcReports, err := source.GCOptimizationDetails(ctx, snapshot, gcDetailsDir)
		if err != nil {
			event.Error(ctx, "warning: gc details", err, tag.Snapshot.Of(snapshot.ID()), tag.Package.Of(pkg.ID()))
		}
		for id, diags := range gcReports {
			fh := snapshot.FindFile(id.URI)
			// Don't publish gc details for unsaved buffers, since the underlying
			// logic operates on the file on disk.
			if fh == nil || !fh.Saved() {
				continue
			}
			s.storeDiagnostics(snapshot, id.URI, gcDetailsSource, diags)
		}
	}
	return shouldWarn(snapshot, pkg)
}

// storeDiagnostics stores results from a single diagnostic source. If merge is
// true, it merges results into any existing results for this snapshot.
func (s *Server) storeDiagnostics(snapshot source.Snapshot, uri span.URI, dsource diagnosticSource, diags []*source.Diagnostic) {
	// Safeguard: ensure that the file actually exists in the snapshot
	// (see golang.org/issues/38602).
	fh := snapshot.FindFile(uri)
	if fh == nil {
		return
	}
	s.diagnosticsMu.Lock()
	defer s.diagnosticsMu.Unlock()
	if s.diagnostics[uri] == nil {
		s.diagnostics[uri] = &fileReports{
			publishedHash: hashDiagnostics(), // Hash for 0 diagnostics.
			reports:       map[diagnosticSource]diagnosticReport{},
		}
	}
	report := s.diagnostics[uri].reports[dsource]
	// Don't set obsolete diagnostics.
	if report.snapshotID > snapshot.ID() {
		return
	}
	if report.diags == nil || report.snapshotID != snapshot.ID() {
		report.diags = map[string]*source.Diagnostic{}
	}
	report.snapshotID = snapshot.ID()
	for _, d := range diags {
		report.diags[hashDiagnostics(d)] = d
	}
	s.diagnostics[uri].reports[dsource] = report
}

// shouldWarn reports whether we should warn the user about their build
// configuration.
func shouldWarn(snapshot source.Snapshot, pkg source.Package) bool {
	if snapshot.ValidBuildConfiguration() {
		return false
	}
	if len(pkg.MissingDependencies()) > 0 {
		return true
	}
	if len(pkg.CompiledGoFiles()) == 1 && hasUndeclaredErrors(pkg) {
		return true // The user likely opened a single file.
	}
	return false
}

// clearDiagnosticSource clears all diagnostics for a given source type. It is
// necessary for cases where diagnostics have been invalidated by something
// other than a snapshot change, for example when gc_details is toggled.
func (s *Server) clearDiagnosticSource(dsource diagnosticSource) {
	s.diagnosticsMu.Lock()
	defer s.diagnosticsMu.Unlock()
	for _, reports := range s.diagnostics {
		delete(reports.reports, dsource)
	}
}

// hasUndeclaredErrors returns true if a package has a type error
// about an undeclared symbol.
//
// TODO(findleyr): switch to using error codes in 1.16
func hasUndeclaredErrors(pkg source.Package) bool {
	for _, err := range pkg.GetErrors() {
		if err.Kind != source.TypeError {
			continue
		}
		if strings.Contains(err.Message, "undeclared name:") {
			return true
		}
	}
	return false
}

// showCriticalErrorStatus shows the error as a progress report.
// If the error is nil, it clears any existing error progress report.
func (s *Server) showCriticalErrorStatus(ctx context.Context, snapshot source.Snapshot, err error) {
	s.criticalErrorStatusMu.Lock()
	defer s.criticalErrorStatusMu.Unlock()

	// Remove all newlines so that the error message can be formatted in a
	// status bar.
	var errMsg string
	if err != nil {
		// Some error messages can also be displayed as diagnostics. But don't
		// show source.ErrorLists as critical errors--only CriticalErrors
		// should be shown.
		if criticalErr := (*source.CriticalError)(nil); errors.As(err, &criticalErr) {
			s.storeErrorDiagnostics(ctx, snapshot, typeCheckSource, criticalErr.ErrorList)
		} else if srcErrList := (source.ErrorList)(nil); errors.As(err, &srcErrList) {
			s.storeErrorDiagnostics(ctx, snapshot, typeCheckSource, srcErrList)
			return
		}
		errMsg = strings.Replace(err.Error(), "\n", " ", -1)
	}

	if s.criticalErrorStatus == nil {
		if errMsg != "" {
			s.criticalErrorStatus = s.progress.start(ctx, "Error loading workspace", errMsg, nil, nil)
		}
		return
	}

	// If an error is already shown to the user, update it or mark it as
	// resolved.
	if errMsg == "" {
		s.criticalErrorStatus.end("Done.")
		s.criticalErrorStatus = nil
	} else {
		s.criticalErrorStatus.report(errMsg, 0)
	}
}

// checkForOrphanedFile checks that the given URIs can be mapped to packages.
// If they cannot and the workspace is not otherwise unloaded, it also surfaces
// a warning, suggesting that the user check the file for build tags.
func (s *Server) checkForOrphanedFile(ctx context.Context, snapshot source.Snapshot, fh source.VersionedFileHandle) *source.Diagnostic {
	if fh.Kind() != source.Go {
		return nil
	}
	pkgs, err := snapshot.PackagesForFile(ctx, fh.URI(), source.TypecheckWorkspace)
	if len(pkgs) > 0 || err == nil {
		return nil
	}
	pgf, err := snapshot.ParseGo(ctx, fh, source.ParseHeader)
	if err != nil {
		return nil
	}
	spn, err := span.NewRange(snapshot.FileSet(), pgf.File.Name.Pos(), pgf.File.Name.End()).Span()
	if err != nil {
		return nil
	}
	rng, err := pgf.Mapper.Range(spn)
	if err != nil {
		return nil
	}
	// TODO(rstambler): We should be able to parse the build tags in the
	// file and show a more specific error message. For now, put the diagnostic
	// on the package declaration.
	return &source.Diagnostic{
		Range: rng,
		Message: fmt.Sprintf(`No packages found for open file %s: %v.
If this file contains build tags, try adding "-tags=<build tag>" to your gopls "buildFlag" configuration (see (https://github.com/golang/tools/blob/master/gopls/doc/settings.md#buildflags-string).
Otherwise, see the troubleshooting guidelines for help investigating (https://github.com/golang/tools/blob/master/gopls/doc/troubleshooting.md).
`, fh.URI().Filename(), err),
		Severity: protocol.SeverityWarning,
		Source:   "compiler",
	}
}

func (s *Server) storeErrorDiagnostics(ctx context.Context, snapshot source.Snapshot, dsource diagnosticSource, errors []*source.Error) {
	for _, e := range errors {
		diagnostic := &source.Diagnostic{
			Range:    e.Range,
			Message:  e.Message,
			Related:  e.Related,
			Severity: protocol.SeverityError,
			Source:   e.Category,
		}
		s.storeDiagnostics(snapshot, e.URI, dsource, []*source.Diagnostic{diagnostic})
	}
}

// publishDiagnostics collects and publishes any unpublished diagnostic reports.
func (s *Server) publishDiagnostics(ctx context.Context, final bool, snapshot source.Snapshot) {
	s.diagnosticsMu.Lock()
	defer s.diagnosticsMu.Unlock()
	for uri, r := range s.diagnostics {
		// Snapshot IDs are always increasing, so we use them instead of file
		// versions to create the correct order for diagnostics.

		// If we've already delivered diagnostics for a future snapshot for this
		// file, do not deliver them.
		if r.snapshotID > snapshot.ID() {
			continue
		}
		anyReportsChanged := false
		reportHashes := map[diagnosticSource]string{}
		var diags []*source.Diagnostic
		for dsource, report := range r.reports {
			if report.snapshotID != snapshot.ID() {
				continue
			}
			var reportDiags []*source.Diagnostic
			for _, d := range report.diags {
				diags = append(diags, d)
				reportDiags = append(reportDiags, d)
			}
			hash := hashDiagnostics(reportDiags...)
			if hash != report.publishedHash {
				anyReportsChanged = true
			}
			reportHashes[dsource] = hash
		}

		if !final && !anyReportsChanged {
			// Don't invalidate existing reports on the client if we haven't got any
			// new information.
			continue
		}
		source.SortDiagnostics(diags)
		hash := hashDiagnostics(diags...)
		if hash == r.publishedHash {
			// Update snapshotID to be the latest snapshot for which this diagnostic
			// hash is valid.
			r.snapshotID = snapshot.ID()
			continue
		}
		version := float64(0)
		if fh := snapshot.FindFile(uri); fh != nil { // file may have been deleted
			version = fh.Version()
		}
		if err := s.client.PublishDiagnostics(ctx, &protocol.PublishDiagnosticsParams{
			Diagnostics: toProtocolDiagnostics(diags),
			URI:         protocol.URIFromSpanURI(uri),
			Version:     version,
		}); err == nil {
			r.publishedHash = hash
			r.snapshotID = snapshot.ID()
			for dsource, hash := range reportHashes {
				report := r.reports[dsource]
				report.publishedHash = hash
				r.reports[dsource] = report
			}
		} else {
			if ctx.Err() != nil {
				// Publish may have failed due to a cancelled context.
				return
			}
			event.Error(ctx, "publishReports: failed to deliver diagnostic", err, tag.URI.Of(uri))
		}
	}
}

func toProtocolDiagnostics(diagnostics []*source.Diagnostic) []protocol.Diagnostic {
	reports := []protocol.Diagnostic{}
	for _, diag := range diagnostics {
		related := make([]protocol.DiagnosticRelatedInformation, 0, len(diag.Related))
		for _, rel := range diag.Related {
			related = append(related, protocol.DiagnosticRelatedInformation{
				Location: protocol.Location{
					URI:   protocol.URIFromSpanURI(rel.URI),
					Range: rel.Range,
				},
				Message: rel.Message,
			})
		}
		reports = append(reports, protocol.Diagnostic{
			// diag.Message might start with \n or \t
			Message:            strings.TrimSpace(diag.Message),
			Range:              diag.Range,
			Severity:           diag.Severity,
			Source:             diag.Source,
			Tags:               diag.Tags,
			RelatedInformation: related,
		})
	}
	return reports
}

func (s *Server) shouldIgnoreError(ctx context.Context, snapshot source.Snapshot, err error) bool {
	if err == nil { // if there is no error at all
		return false
	}
	if errors.Is(err, context.Canceled) {
		return true
	}
	// If the folder has no Go code in it, we shouldn't spam the user with a warning.
	var hasGo bool
	_ = filepath.Walk(snapshot.View().Folder().Filename(), func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !strings.HasSuffix(info.Name(), ".go") {
			return nil
		}
		hasGo = true
		return errors.New("done")
	})
	return !hasGo
}
