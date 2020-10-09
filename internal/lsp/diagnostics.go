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

// idWithAnalysis is used to track if the diagnostics for a given file were
// computed with analyses.
type idWithAnalysis struct {
	id           source.VersionedFileIdentity
	withAnalysis bool
}

// A reportSet collects diagnostics for publication, sorting them by file and
// de-duplicating.
type reportSet struct {
	mu sync.Mutex
	// lazily allocated
	reports map[idWithAnalysis]map[string]*source.Diagnostic
}

func (s *reportSet) add(id source.VersionedFileIdentity, withAnalysis bool, diags ...*source.Diagnostic) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.reports == nil {
		s.reports = make(map[idWithAnalysis]map[string]*source.Diagnostic)
	}
	key := idWithAnalysis{
		id:           id,
		withAnalysis: withAnalysis,
	}
	if _, ok := s.reports[key]; !ok {
		s.reports[key] = map[string]*source.Diagnostic{}
	}
	for _, d := range diags {
		s.reports[key][diagnosticKey(d)] = d
	}
}

// diagnosticKey creates a unique identifier for a given diagnostic, since we
// cannot use source.Diagnostics as map keys. This is used to de-duplicate
// diagnostics.
func diagnosticKey(d *source.Diagnostic) string {
	var tags, related string
	for _, t := range d.Tags {
		tags += fmt.Sprintf("%s", t)
	}
	for _, r := range d.Related {
		related += fmt.Sprintf("%s%s%s", r.URI, r.Message, r.Range)
	}
	key := fmt.Sprintf("%s%s%s%s%s%s", d.Message, d.Range, d.Severity, d.Source, tags, related)
	return fmt.Sprintf("%x", sha256.Sum256([]byte(key)))
}

func (s *Server) diagnoseDetached(snapshot source.Snapshot) {
	ctx := snapshot.View().BackgroundContext()
	ctx = xcontext.Detach(ctx)
	reports, shows := s.diagnose(ctx, snapshot, false)
	if shows != nil {
		// If a view has been created or the configuration changed, warn the user.
		s.client.ShowMessage(ctx, shows)
	}
	s.publishReports(ctx, snapshot, reports)
}

func (s *Server) diagnoseSnapshot(snapshot source.Snapshot, changedURIs []span.URI) {
	ctx := snapshot.View().BackgroundContext()

	delay := snapshot.View().Options().ExperimentalDiagnosticsDelay
	if delay > 0 {
		// Experimental 2-phase diagnostics.
		//
		// The first phase just parses and checks packages that have been affected
		// by file modifications (no analysis).
		//
		// The second phase does everything, and is debounced by the configured delay.
		reports, err := s.diagnoseChangedFiles(ctx, snapshot, changedURIs)
		if err != nil {
			if !errors.Is(err, context.Canceled) {
				event.Error(ctx, "diagnosing changed files", err)
			}
		}
		s.publishReports(ctx, snapshot, reports)
		s.debouncer.debounce(snapshot.View().Name(), snapshot.ID(), delay, func() {
			reports, _ := s.diagnose(ctx, snapshot, false)
			s.publishReports(ctx, snapshot, reports)
		})
		return
	}

	// Ignore possible workspace configuration warnings in the normal flow.
	reports, _ := s.diagnose(ctx, snapshot, false)
	s.publishReports(ctx, snapshot, reports)
}

func (s *Server) diagnoseChangedFiles(ctx context.Context, snapshot source.Snapshot, uris []span.URI) (*reportSet, error) {
	ctx, done := event.Start(ctx, "Server.diagnoseChangedFiles")
	defer done()
	packages := make(map[source.Package]struct{})
	for _, uri := range uris {
		pkgs, err := snapshot.PackagesForFile(ctx, uri, source.TypecheckWorkspace)
		if err != nil {
			// TODO (rFindley): we should probably do something with the error here,
			// but as of now this can fail repeatedly if load fails, so can be too
			// noisy to log (and we'll handle things later in the slow pass).
			continue
		}
		for _, pkg := range pkgs {
			packages[pkg] = struct{}{}
		}
	}
	reports := new(reportSet)
	for pkg := range packages {
		pkgReports, _, err := source.Diagnostics(ctx, snapshot, pkg, false)
		if err != nil {
			return nil, err
		}
		for id, diags := range pkgReports {
			reports.add(id, false, diags...)
		}
	}
	return reports, nil
}

// diagnose is a helper function for running diagnostics with a given context.
// Do not call it directly.
func (s *Server) diagnose(ctx context.Context, snapshot source.Snapshot, alwaysAnalyze bool) (diagReports *reportSet, _ *protocol.ShowMessageParams) {
	ctx, done := event.Start(ctx, "Server.diagnose")
	defer done()

	// Wait for a free diagnostics slot.
	select {
	case <-ctx.Done():
		return nil, nil
	case s.diagnosticsSema <- struct{}{}:
	}
	defer func() {
		<-s.diagnosticsSema
	}()

	reports := new(reportSet)

	// First, diagnose the go.mod file.
	modReports, modErr := mod.Diagnostics(ctx, snapshot)
	if ctx.Err() != nil {
		return nil, nil
	}
	if modErr != nil {
		event.Error(ctx, "warning: diagnose go.mod", modErr, tag.Directory.Of(snapshot.View().Folder().Filename()), tag.Snapshot.Of(snapshot.ID()))
	}
	for id, diags := range modReports {
		if id.URI == "" {
			event.Error(ctx, "missing URI for module diagnostics", fmt.Errorf("empty URI"), tag.Directory.Of(snapshot.View().Folder().Filename()))
			continue
		}
		reports.add(id, true, diags...) // treat go.mod diagnostics like analyses
	}

	// Diagnose all of the packages in the workspace.
	wsPkgs, err := snapshot.WorkspacePackages(ctx)
	if err != nil {
		if errors.Is(err, context.Canceled) {
			return nil, nil
		}
		// Some error messages can be displayed as diagnostics.
		if errList := (*source.ErrorList)(nil); errors.As(err, &errList) {
			if err := errorsToDiagnostic(ctx, snapshot, *errList, reports); err == nil {
				return reports, nil
			}
		}
		// Try constructing a more helpful error message out of this error.
		if s.handleFatalErrors(ctx, snapshot, modErr, err) {
			return nil, nil
		}
		event.Error(ctx, "errors diagnosing workspace", err, tag.Snapshot.Of(snapshot.ID()), tag.Directory.Of(snapshot.View().Folder()))
		// Present any `go list` errors directly to the user.
		if errors.Is(err, source.PackagesLoadError) {
			if err := s.client.ShowMessage(ctx, &protocol.ShowMessageParams{
				Type: protocol.Error,
				Message: fmt.Sprintf(`The code in the workspace failed to compile (see the error message below).
If you believe this is a mistake, please file an issue: https://github.com/golang/go/issues/new.
%v`, err),
			}); err != nil {
				event.Error(ctx, "ShowMessage failed", err, tag.Directory.Of(snapshot.View().Folder().Filename()))
			}
		}
		return nil, nil
	}
	var (
		showMsg *protocol.ShowMessageParams
		wg      sync.WaitGroup
	)
	for _, pkg := range wsPkgs {
		wg.Add(1)
		go func(pkg source.Package) {
			defer wg.Done()

			withAnalysis := alwaysAnalyze // only run analyses for packages with open files
			var gcDetailsDir span.URI     // find the package's optimization details, if available
			for _, pgf := range pkg.CompiledGoFiles() {
				if snapshot.IsOpen(pgf.URI) {
					withAnalysis = true
				}
				if gcDetailsDir == "" {
					dirURI := span.URIFromPath(filepath.Dir(pgf.URI.Filename()))
					s.gcOptimizationDetailsMu.Lock()
					_, ok := s.gcOptimizatonDetails[dirURI]
					s.gcOptimizationDetailsMu.Unlock()
					if ok {
						gcDetailsDir = dirURI
					}
				}
			}

			pkgReports, warn, err := source.Diagnostics(ctx, snapshot, pkg, withAnalysis)

			// Check if might want to warn the user about their build configuration.
			// Our caller decides whether to send the message.
			if warn && !snapshot.ValidBuildConfiguration() {
				showMsg = &protocol.ShowMessageParams{
					Type:    protocol.Warning,
					Message: `You are neither in a module nor in your GOPATH. If you are using modules, please open your editor to a directory in your module. If you believe this warning is incorrect, please file an issue: https://github.com/golang/go/issues/new.`,
				}
			}
			if err != nil {
				event.Error(ctx, "warning: diagnose package", err, tag.Snapshot.Of(snapshot.ID()), tag.Package.Of(pkg.ID()))
				return
			}

			// Add all reports to the global map, checking for duplicates.
			for id, diags := range pkgReports {
				reports.add(id, withAnalysis, diags...)
			}
			// If gc optimization details are available, add them to the
			// diagnostic reports.
			if gcDetailsDir != "" {
				gcReports, err := source.GCOptimizationDetails(ctx, snapshot, gcDetailsDir)
				if err != nil {
					event.Error(ctx, "warning: gc details", err, tag.Snapshot.Of(snapshot.ID()))
				}
				for id, diags := range gcReports {
					reports.add(id, withAnalysis, diags...)
				}
			}
		}(pkg)
	}
	wg.Wait()
	// Confirm that every opened file belongs to a package (if any exist in
	// the workspace). Otherwise, add a diagnostic to the file.
	if len(wsPkgs) > 0 {
		for _, o := range s.session.Overlays() {
			// Check if we already have diagnostic reports for the given file,
			// meaning that we have already seen its package.
			var seen bool
			for _, withAnalysis := range []bool{true, false} {
				_, ok := reports.reports[idWithAnalysis{
					id:           o.VersionedFileIdentity(),
					withAnalysis: withAnalysis,
				}]
				seen = seen || ok
			}
			if seen {
				continue
			}
			diagnostic := s.checkForOrphanedFile(ctx, snapshot, o)
			if diagnostic == nil {
				continue
			}
			reports.add(o.VersionedFileIdentity(), true, diagnostic)
		}
	}
	return reports, showMsg
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

func errorsToDiagnostic(ctx context.Context, snapshot source.Snapshot, errors []*source.Error, reports *reportSet) error {
	for _, e := range errors {
		diagnostic := &source.Diagnostic{
			Range:    e.Range,
			Message:  e.Message,
			Related:  e.Related,
			Severity: protocol.SeverityError,
			Source:   e.Category,
		}
		fh, err := snapshot.GetFile(ctx, e.URI)
		if err != nil {
			return err
		}
		reports.add(fh.VersionedFileIdentity(), false, diagnostic)
	}
	return nil
}

func (s *Server) publishReports(ctx context.Context, snapshot source.Snapshot, reports *reportSet) {
	// Check for context cancellation before publishing diagnostics.
	if ctx.Err() != nil || reports == nil {
		return
	}

	s.deliveredMu.Lock()
	defer s.deliveredMu.Unlock()

	for key, diagnosticsMap := range reports.reports {
		// Don't deliver diagnostics if the context has already been canceled.
		if ctx.Err() != nil {
			break
		}
		// Pre-sort diagnostics to avoid extra work when we compare them.
		var diagnostics []*source.Diagnostic
		for _, d := range diagnosticsMap {
			diagnostics = append(diagnostics, d)
		}
		source.SortDiagnostics(diagnostics)
		toSend := sentDiagnostics{
			id:           key.id,
			sorted:       diagnostics,
			withAnalysis: key.withAnalysis,
			snapshotID:   snapshot.ID(),
		}

		// We use the zero values if this is an unknown file.
		delivered := s.delivered[key.id.URI]

		// Snapshot IDs are always increasing, so we use them instead of file
		// versions to create the correct order for diagnostics.

		// If we've already delivered diagnostics for a future snapshot for this file,
		// do not deliver them.
		if delivered.snapshotID > toSend.snapshotID {
			// Do not update the delivered map since it already contains newer diagnostics.
			continue
		}

		// Check if we should reuse the cached diagnostics.
		if equalDiagnostics(delivered.sorted, diagnostics) {
			// Make sure to update the delivered map.
			s.delivered[key.id.URI] = toSend
			continue
		}

		// If we've already delivered diagnostics for this file, at this
		// snapshot, with analyses, do not send diagnostics without analyses.
		if delivered.snapshotID == toSend.snapshotID && delivered.id == toSend.id &&
			delivered.withAnalysis && !toSend.withAnalysis {
			// Do not update the delivered map since it already contains better diagnostics.
			continue
		}
		if err := s.client.PublishDiagnostics(ctx, &protocol.PublishDiagnosticsParams{
			Diagnostics: toProtocolDiagnostics(diagnostics),
			URI:         protocol.URIFromSpanURI(key.id.URI),
			Version:     key.id.Version,
		}); err != nil {
			event.Error(ctx, "publishReports: failed to deliver diagnostic", err, tag.URI.Of(key.id.URI))
			continue
		}
		// Update the delivered map.
		s.delivered[key.id.URI] = toSend
	}
}

// equalDiagnostics returns true if the 2 lists of diagnostics are equal.
// It assumes that both a and b are already sorted.
func equalDiagnostics(a, b []*source.Diagnostic) bool {
	if len(a) != len(b) {
		return false
	}
	for i := 0; i < len(a); i++ {
		if source.CompareDiagnostic(a[i], b[i]) != 0 {
			return false
		}
	}
	return true
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

func (s *Server) handleFatalErrors(ctx context.Context, snapshot source.Snapshot, modErr, loadErr error) bool {
	// If the folder has no Go code in it, we shouldn't spam the user with a warning.
	var hasGo bool
	_ = filepath.Walk(snapshot.View().Folder().Filename(), func(path string, info os.FileInfo, err error) error {
		if !strings.HasSuffix(info.Name(), ".go") {
			return nil
		}
		hasGo = true
		return errors.New("done")
	})
	if !hasGo {
		return true
	}

	// All other workarounds are for errors associated with modules.
	if len(snapshot.ModFiles()) == 0 {
		return false
	}

	switch loadErr {
	case source.InconsistentVendoring:
		item, err := s.client.ShowMessageRequest(ctx, &protocol.ShowMessageRequestParams{
			Type: protocol.Error,
			Message: `Inconsistent vendoring detected. Please re-run "go mod vendor".
See https://github.com/golang/go/issues/39164 for more detail on this issue.`,
			Actions: []protocol.MessageActionItem{
				{Title: "go mod vendor"},
			},
		})
		// If the user closes the pop-up, don't show them further errors.
		if item == nil {
			return true
		}
		if err != nil {
			event.Error(ctx, "go mod vendor ShowMessageRequest failed", err, tag.Directory.Of(snapshot.View().Folder().Filename()))
			return true
		}
		// Right now, we don't have a good way of mapping the error message
		// to a specific module, so this will re-run `go mod vendor` in every
		// known module with a vendor directory.
		// TODO(rstambler): Only re-run `go mod vendor` in the relevant module.
		for _, uri := range snapshot.ModFiles() {
			// Check that there is a vendor directory in the module before
			// running `go mod vendor`.
			if info, _ := os.Stat(filepath.Join(filepath.Dir(uri.Filename()), "vendor")); info == nil {
				continue
			}
			if err := s.directGoModCommand(ctx, protocol.URIFromSpanURI(uri), "mod", []string{"vendor"}...); err != nil {
				if err := s.client.ShowMessage(ctx, &protocol.ShowMessageParams{
					Type:    protocol.Error,
					Message: fmt.Sprintf(`"go mod vendor" failed with %v`, err),
				}); err != nil {
					if err != nil {
						event.Error(ctx, "go mod vendor ShowMessage failed", err, tag.Directory.Of(snapshot.View().Folder().Filename()))
					}
				}
			}
		}
		return true
	}
	return false
}
