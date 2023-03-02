// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"golang.org/x/sync/errgroup"
	"golang.org/x/tools/gopls/internal/lsp/debug/log"
	"golang.org/x/tools/gopls/internal/lsp/mod"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/lsp/template"
	"golang.org/x/tools/gopls/internal/lsp/work"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/event/tag"
	"golang.org/x/tools/internal/xcontext"
)

// diagnosticSource differentiates different sources of diagnostics.
type diagnosticSource int

const (
	modSource diagnosticSource = iota
	gcDetailsSource
	analysisSource
	typeCheckSource
	orphanedSource
	workSource
	modCheckUpgradesSource
	modVulncheckSource // source.Govulncheck + source.Vulncheck
)

// A diagnosticReport holds results for a single diagnostic source.
type diagnosticReport struct {
	snapshotID    source.GlobalSnapshotID // global snapshot ID on which the report was computed
	publishedHash string                  // last published hash for this (URI, source)
	diags         map[string]*source.Diagnostic
}

// fileReports holds a collection of diagnostic reports for a single file, as
// well as the hash of the last published set of diagnostics.
type fileReports struct {
	// publishedSnapshotID is the last snapshot ID for which we have "published"
	// diagnostics (though the publishDiagnostics notification may not have
	// actually been sent, if nothing changed).
	//
	// Specifically, publishedSnapshotID is updated to a later snapshot ID when
	// we either:
	//  (1) publish diagnostics for the file for a snapshot, or
	//  (2) determine that published diagnostics are valid for a new snapshot.
	//
	// Notably publishedSnapshotID may not match the snapshot id on individual reports in
	// the reports map:
	// - we may have published partial diagnostics from only a subset of
	//   diagnostic sources for which new results have been computed, or
	// - we may have started computing reports for an even new snapshot, but not
	//   yet published.
	//
	// This prevents gopls from publishing stale diagnostics.
	publishedSnapshotID source.GlobalSnapshotID

	// publishedHash is a hash of the latest diagnostics published for the file.
	publishedHash string

	// If set, mustPublish marks diagnostics as needing publication, independent
	// of whether their publishedHash has changed.
	mustPublish bool

	// The last stored diagnostics for each diagnostic source.
	reports map[diagnosticSource]diagnosticReport
}

func (d diagnosticSource) String() string {
	switch d {
	case modSource:
		return "FromSource"
	case gcDetailsSource:
		return "FromGCDetails"
	case analysisSource:
		return "FromAnalysis"
	case typeCheckSource:
		return "FromTypeChecking"
	case orphanedSource:
		return "FromOrphans"
	case workSource:
		return "FromGoWork"
	case modCheckUpgradesSource:
		return "FromCheckForUpgrades"
	case modVulncheckSource:
		return "FromModVulncheck"
	default:
		return fmt.Sprintf("From?%d?", d)
	}
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
			fmt.Fprintf(h, "%s%s%s", r.Location.URI.SpanURI(), r.Message, r.Location.Range)
		}
		fmt.Fprintf(h, "%s%s%s%s", d.Message, d.Range, d.Severity, d.Source)
	}
	return fmt.Sprintf("%x", h.Sum(nil))
}

func (s *Server) diagnoseDetached(snapshot source.Snapshot) {
	ctx := snapshot.BackgroundContext()
	ctx = xcontext.Detach(ctx)
	s.diagnose(ctx, snapshot, false)
	s.publishDiagnostics(ctx, true, snapshot)
}

func (s *Server) diagnoseSnapshots(snapshots map[source.Snapshot][]span.URI, onDisk bool) {
	var diagnosticWG sync.WaitGroup
	for snapshot, uris := range snapshots {
		diagnosticWG.Add(1)
		go func(snapshot source.Snapshot, uris []span.URI) {
			defer diagnosticWG.Done()
			s.diagnoseSnapshot(snapshot, uris, onDisk)
		}(snapshot, uris)
	}
	diagnosticWG.Wait()
}

func (s *Server) diagnoseSnapshot(snapshot source.Snapshot, changedURIs []span.URI, onDisk bool) {
	ctx := snapshot.BackgroundContext()
	ctx, done := event.Start(ctx, "Server.diagnoseSnapshot", source.SnapshotLabels(snapshot)...)
	defer done()

	delay := snapshot.View().Options().DiagnosticsDelay
	if delay > 0 {
		// 2-phase diagnostics.
		//
		// The first phase just parses and type-checks (but
		// does not analyze) packages directly affected by
		// file modifications.
		//
		// The second phase runs analysis on the entire snapshot,
		// and is debounced by the configured delay.
		s.diagnoseChangedFiles(ctx, snapshot, changedURIs, onDisk)
		s.publishDiagnostics(ctx, false, snapshot)

		// We debounce diagnostics separately for each view, using the snapshot
		// local ID as logical ordering.
		//
		// TODO(rfindley): it would be cleaner to simply put the diagnostic
		// debouncer on the view, and remove the "key" argument to debouncing.
		if ok := <-s.diagDebouncer.debounce(snapshot.View().Name(), snapshot.SequenceID(), time.After(delay)); ok {
			s.diagnose(ctx, snapshot, false)
			s.publishDiagnostics(ctx, true, snapshot)
		}
		return
	}

	// Ignore possible workspace configuration warnings in the normal flow.
	s.diagnose(ctx, snapshot, false)
	s.publishDiagnostics(ctx, true, snapshot)
}

func (s *Server) diagnoseChangedFiles(ctx context.Context, snapshot source.Snapshot, uris []span.URI, onDisk bool) {
	ctx, done := event.Start(ctx, "Server.diagnoseChangedFiles", source.SnapshotLabels(snapshot)...)
	defer done()

	seen := make(map[source.PackageID]bool)
	var toDiagnose []*source.Metadata
	for _, uri := range uris {
		// If the change is only on-disk and the file is not open, don't
		// directly request its package. It may not be a workspace package.
		if onDisk && !snapshot.IsOpen(uri) {
			continue
		}
		// If the file is not known to the snapshot (e.g., if it was deleted),
		// don't diagnose it.
		if snapshot.FindFile(uri) == nil {
			continue
		}

		// Don't request type-checking for builtin.go: it's not a real package.
		if snapshot.IsBuiltin(ctx, uri) {
			continue
		}

		// Find all packages that include this file and diagnose them in parallel.
		metas, err := snapshot.MetadataForFile(ctx, uri)
		if err != nil {
			if ctx.Err() != nil {
				return
			}
			// TODO(findleyr): we should probably do something with the error here,
			// but as of now this can fail repeatedly if load fails, so can be too
			// noisy to log (and we'll handle things later in the slow pass).
			continue
		}
		for _, m := range metas {
			if m.IsIntermediateTestVariant() {
				continue
			}
			if !seen[m.ID] {
				seen[m.ID] = true
				toDiagnose = append(toDiagnose, m)
			}
		}
	}
	s.diagnosePkgs(ctx, snapshot, toDiagnose, false)
}

// diagnose is a helper function for running diagnostics with a given context.
// Do not call it directly. forceAnalysis is only true for testing purposes.
func (s *Server) diagnose(ctx context.Context, snapshot source.Snapshot, forceAnalysis bool) {
	ctx, done := event.Start(ctx, "Server.diagnose", source.SnapshotLabels(snapshot)...)
	defer done()

	// Wait for a free diagnostics slot.
	// TODO(adonovan): opt: shouldn't it be the analysis implementation's
	// job to de-dup and limit resource consumption? In any case this
	// this function spends most its time waiting for awaitLoaded, at
	// least initially.
	select {
	case <-ctx.Done():
		return
	case s.diagnosticsSema <- struct{}{}:
	}
	defer func() {
		<-s.diagnosticsSema
	}()

	// common code for dispatching diagnostics
	store := func(dsource diagnosticSource, operation string, diagsByFile map[span.URI][]*source.Diagnostic, err error, merge bool) {
		if err != nil {
			event.Error(ctx, "warning: while "+operation, err, source.SnapshotLabels(snapshot)...)
		}
		for uri, diags := range diagsByFile {
			if uri == "" {
				event.Error(ctx, "missing URI while "+operation, fmt.Errorf("empty URI"), tag.Directory.Of(snapshot.View().Folder().Filename()))
				continue
			}
			s.storeDiagnostics(snapshot, uri, dsource, diags, merge)
		}
	}

	// Diagnose go.mod upgrades.
	upgradeReports, upgradeErr := mod.UpgradeDiagnostics(ctx, snapshot)
	if ctx.Err() != nil {
		log.Trace.Log(ctx, "diagnose cancelled")
		return
	}
	store(modCheckUpgradesSource, "diagnosing go.mod upgrades", upgradeReports, upgradeErr, true)

	// Diagnose go.work file.
	workReports, workErr := work.Diagnostics(ctx, snapshot)
	if ctx.Err() != nil {
		log.Trace.Log(ctx, "diagnose cancelled")
		return
	}
	store(workSource, "diagnosing go.work file", workReports, workErr, true)

	// Diagnose go.mod file.
	// (This step demands type checking of all active packages:
	// the bottleneck in the startup sequence for a big workspace.)
	modReports, modErr := mod.Diagnostics(ctx, snapshot)
	if ctx.Err() != nil {
		log.Trace.Log(ctx, "diagnose cancelled")
		return
	}
	store(modSource, "diagnosing go.mod file", modReports, modErr, true)

	// Diagnose vulnerabilities.
	vulnReports, vulnErr := mod.VulnerabilityDiagnostics(ctx, snapshot)
	if ctx.Err() != nil {
		log.Trace.Log(ctx, "diagnose cancelled")
		return
	}
	store(modVulncheckSource, "diagnosing vulnerabilities", vulnReports, vulnErr, false)

	activeMetas, activeErr := snapshot.ActiveMetadata(ctx)
	if s.shouldIgnoreError(ctx, snapshot, activeErr) {
		return
	}
	criticalErr := snapshot.GetCriticalError(ctx)
	if ctx.Err() != nil { // must check ctx after GetCriticalError
		return
	}

	// Show the error as a progress error report so that it appears in the
	// status bar. If a client doesn't support progress reports, the error
	// will still be shown as a ShowMessage. If there is no error, any running
	// error progress reports will be closed.
	s.showCriticalErrorStatus(ctx, snapshot, criticalErr)

	// Diagnose template (.tmpl) files.
	for _, f := range snapshot.Templates() {
		diags := template.Diagnose(f)
		s.storeDiagnostics(snapshot, f.URI(), typeCheckSource, diags, true)
	}

	// If there are no workspace packages, there is nothing to diagnose and
	// there are no orphaned files.
	if len(activeMetas) == 0 {
		return
	}

	// Run go/analysis diagnosis of packages in parallel.
	// TODO(adonovan): opt: it may be more efficient to
	// have diagnosePkg take a set of packages.
	//
	// TODO(adonovan): opt: since the new analysis driver does its
	// own type checking, we could strength-reduce pkg to
	// PackageID and get this step started as soon as the set of
	// active package IDs are known, without waiting for them to load.
	var (
		seen       = map[span.URI]struct{}{}
		toDiagnose []*source.Metadata
	)
	for _, m := range activeMetas {
		for _, uri := range m.CompiledGoFiles {
			seen[uri] = struct{}{}
		}
		toDiagnose = append(toDiagnose, m)
	}
	s.diagnosePkgs(ctx, snapshot, toDiagnose, forceAnalysis)

	// Orphaned files.
	// Confirm that every opened file belongs to a package (if any exist in
	// the workspace). Otherwise, add a diagnostic to the file.
	for _, o := range s.session.Overlays() {
		if _, ok := seen[o.URI()]; ok {
			continue
		}
		diagnostic := s.checkForOrphanedFile(ctx, snapshot, o)
		if diagnostic == nil {
			continue
		}
		s.storeDiagnostics(snapshot, o.URI(), orphanedSource, []*source.Diagnostic{diagnostic}, true)
	}
}

func (s *Server) diagnosePkgs(ctx context.Context, snapshot source.Snapshot, mds []*source.Metadata, alwaysAnalyze bool) {
	ctx, done := event.Start(ctx, "Server.diagnosePkgs", source.SnapshotLabels(snapshot)...)
	defer done()
	var needIDs []source.PackageID // exclude e.g. testdata packages
	needAnalysis := make(map[source.PackageID]*source.Metadata)
	for _, m := range mds {
		enableDiagnostics := false
		includeAnalysis := alwaysAnalyze // only run analyses for packages with open files
		for _, uri := range m.CompiledGoFiles {
			enableDiagnostics = enableDiagnostics || !snapshot.IgnoredFile(uri)
			includeAnalysis = includeAnalysis || snapshot.IsOpen(uri)
		}
		if enableDiagnostics {
			needIDs = append(needIDs, m.ID)
		}
		if includeAnalysis && enableDiagnostics {
			needAnalysis[m.ID] = m
		}
	}

	// Analyze and type-check concurrently, since they are independent
	// operations.
	var (
		g             errgroup.Group
		diags         map[span.URI][]*source.Diagnostic
		analysisMu    sync.Mutex
		analysisDiags = make(map[span.URI][]*source.Diagnostic)
	)

	// Collect package diagnostics.
	g.Go(func() error {
		var err error
		diags, err = snapshot.PackageDiagnostics(ctx, needIDs...)
		if err != nil {
			event.Error(ctx, "warning: diagnostics failed", err, source.SnapshotLabels(snapshot)...)
		}
		return err
	})

	// Get diagnostics from analysis framework.
	// This includes type-error analyzers, which suggest fixes to compiler errors.
	for _, m := range needAnalysis {
		m := m
		g.Go(func() error {
			diags, err := source.Analyze(ctx, snapshot, m.ID, false)
			if err != nil {
				event.Error(ctx, "warning: analyzing package", err, append(source.SnapshotLabels(snapshot), tag.Package.Of(string(m.ID)))...)
				return nil
			}
			analysisMu.Lock()
			for uri, diags := range diags {
				analysisDiags[uri] = append(analysisDiags[uri], diags...)
			}
			analysisMu.Unlock()
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return // error handled elsewhere
	}

	for _, m := range mds {
		// For each file, update the server's diagnostics state.
		for _, uri := range m.CompiledGoFiles {
			// builtin.go exists only for documentation purposes and
			// is not valid Go code. Don't report distracting errors.
			if snapshot.IsBuiltin(ctx, uri) {
				continue
			}

			pkgDiags := diags[uri]
			var tdiags, adiags []*source.Diagnostic
			source.CombineDiagnostics(pkgDiags, analysisDiags[uri], &tdiags, &adiags)
			s.storeDiagnostics(snapshot, uri, typeCheckSource, tdiags, true)
			s.storeDiagnostics(snapshot, uri, analysisSource, adiags, true)
		}

		// If gc optimization details are requested, add them to the
		// diagnostic reports.
		s.gcOptimizationDetailsMu.Lock()
		_, enableGCDetails := s.gcOptimizationDetails[m.ID]
		s.gcOptimizationDetailsMu.Unlock()

		// TODO(rfindley): we should avoid running the go command repeatedly here.
		if enableGCDetails {
			gcReports, err := source.GCOptimizationDetails(ctx, snapshot, m)
			if err != nil {
				event.Error(ctx, "warning: gc details", err, append(source.SnapshotLabels(snapshot), tag.Package.Of(string(m.ID)))...)
			}
			s.gcOptimizationDetailsMu.Lock()
			_, enableGCDetails := s.gcOptimizationDetails[m.ID]

			// NOTE(golang/go#44826): hold the gcOptimizationDetails lock, and re-check
			// whether gc optimization details are enabled, while storing gc_details
			// results. This ensures that the toggling of GC details and clearing of
			// diagnostics does not race with storing the results here.
			if enableGCDetails {
				for uri, diags := range gcReports {
					fh := snapshot.FindFile(uri)
					// Don't publish gc details for unsaved buffers, since the underlying
					// logic operates on the file on disk.
					if fh == nil || !fh.Saved() {
						continue
					}
					s.storeDiagnostics(snapshot, uri, gcDetailsSource, diags, true)
				}
			}
			s.gcOptimizationDetailsMu.Unlock()
		}
	}
}

// mustPublishDiagnostics marks the uri as needing publication, independent of
// whether the published contents have changed.
//
// This can be used for ensuring gopls publishes diagnostics after certain file
// events.
func (s *Server) mustPublishDiagnostics(uri span.URI) {
	s.diagnosticsMu.Lock()
	defer s.diagnosticsMu.Unlock()

	if s.diagnostics[uri] == nil {
		s.diagnostics[uri] = &fileReports{
			publishedHash: hashDiagnostics(), // Hash for 0 diagnostics.
			reports:       map[diagnosticSource]diagnosticReport{},
		}
	}
	s.diagnostics[uri].mustPublish = true
}

// storeDiagnostics stores results from a single diagnostic source. If merge is
// true, it merges results into any existing results for this snapshot.
//
// TODO(hyangah): investigate whether we can unconditionally overwrite previous report.diags
// with the new diags and eliminate the need for the `merge` flag.
func (s *Server) storeDiagnostics(snapshot source.Snapshot, uri span.URI, dsource diagnosticSource, diags []*source.Diagnostic, merge bool) {
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
	if report.snapshotID > snapshot.GlobalID() {
		return
	}
	if report.diags == nil || report.snapshotID != snapshot.GlobalID() || !merge {
		report.diags = map[string]*source.Diagnostic{}
	}
	report.snapshotID = snapshot.GlobalID()
	for _, d := range diags {
		report.diags[hashDiagnostics(d)] = d
	}
	s.diagnostics[uri].reports[dsource] = report
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

const WorkspaceLoadFailure = "Error loading workspace"

// showCriticalErrorStatus shows the error as a progress report.
// If the error is nil, it clears any existing error progress report.
func (s *Server) showCriticalErrorStatus(ctx context.Context, snapshot source.Snapshot, err *source.CriticalError) {
	s.criticalErrorStatusMu.Lock()
	defer s.criticalErrorStatusMu.Unlock()

	// Remove all newlines so that the error message can be formatted in a
	// status bar.
	var errMsg string
	if err != nil {
		event.Error(ctx, "errors loading workspace", err.MainError, source.SnapshotLabels(snapshot)...)
		for _, d := range err.Diagnostics {
			s.storeDiagnostics(snapshot, d.URI, modSource, []*source.Diagnostic{d}, true)
		}
		errMsg = strings.ReplaceAll(err.MainError.Error(), "\n", " ")
	}

	if s.criticalErrorStatus == nil {
		if errMsg != "" {
			s.criticalErrorStatus = s.progress.Start(ctx, WorkspaceLoadFailure, errMsg, nil, nil)
		}
		return
	}

	// If an error is already shown to the user, update it or mark it as
	// resolved.
	if errMsg == "" {
		s.criticalErrorStatus.End(ctx, "Done.")
		s.criticalErrorStatus = nil
	} else {
		s.criticalErrorStatus.Report(ctx, errMsg, 0)
	}
}

// checkForOrphanedFile checks that the given URIs can be mapped to packages.
// If they cannot and the workspace is not otherwise unloaded, it also surfaces
// a warning, suggesting that the user check the file for build tags.
func (s *Server) checkForOrphanedFile(ctx context.Context, snapshot source.Snapshot, fh source.FileHandle) *source.Diagnostic {
	// TODO(rfindley): this function may fail to produce a diagnostic for a
	// variety of reasons, some of which should probably not be ignored. For
	// example, should this function be tolerant of the case where fh does not
	// exist, or does not have a package name?
	//
	// It would be better to panic or report a bug in several of the cases below,
	// so that we can move toward guaranteeing we show the user a meaningful
	// error whenever it makes sense.
	if snapshot.View().FileKind(fh) != source.Go {
		return nil
	}
	// builtin files won't have a package, but they are never orphaned.
	if snapshot.IsBuiltin(ctx, fh.URI()) {
		return nil
	}

	// This call has the effect of inserting fh into snapshot.files,
	// where for better or worse (actually: just worse) it influences
	// the sets of open, known, and orphaned files.
	snapshot.GetFile(ctx, fh.URI())

	metas, _ := snapshot.MetadataForFile(ctx, fh.URI())
	if len(metas) > 0 || ctx.Err() != nil {
		return nil // no package, or cancelled
	}
	// Inv: file does not belong to a package we know about.
	pgf, err := snapshot.ParseGo(ctx, fh, source.ParseHeader)
	if err != nil {
		return nil
	}
	if !pgf.File.Name.Pos().IsValid() {
		return nil
	}
	rng, err := pgf.NodeRange(pgf.File.Name)
	if err != nil {
		return nil
	}
	// If the file no longer has a name ending in .go, this diagnostic is wrong
	if filepath.Ext(fh.URI().Filename()) != ".go" {
		return nil
	}
	// TODO(rstambler): We should be able to parse the build tags in the
	// file and show a more specific error message. For now, put the diagnostic
	// on the package declaration.
	return &source.Diagnostic{
		URI:      fh.URI(),
		Range:    rng,
		Severity: protocol.SeverityWarning,
		Source:   source.ListError,
		Message: fmt.Sprintf(`No packages found for open file %s: %v.
If this file contains build tags, try adding "-tags=<build tag>" to your gopls "buildFlags" configuration (see (https://github.com/golang/tools/blob/master/gopls/doc/settings.md#buildflags-string).
Otherwise, see the troubleshooting guidelines for help investigating (https://github.com/golang/tools/blob/master/gopls/doc/troubleshooting.md).
`, fh.URI().Filename(), err),
	}
}

// publishDiagnostics collects and publishes any unpublished diagnostic reports.
func (s *Server) publishDiagnostics(ctx context.Context, final bool, snapshot source.Snapshot) {
	ctx, done := event.Start(ctx, "Server.publishDiagnostics", source.SnapshotLabels(snapshot)...)
	defer done()

	s.diagnosticsMu.Lock()
	defer s.diagnosticsMu.Unlock()

	for uri, r := range s.diagnostics {
		// Global snapshot IDs are monotonic, so we use them to enforce an ordering
		// for diagnostics.
		//
		// If we've already delivered diagnostics for a future snapshot for this
		// file, do not deliver them. See golang/go#42837 for an example of why
		// this is necessary.
		//
		// TODO(rfindley): even using a global snapshot ID, this mechanism is
		// potentially racy: elsewhere in the code (e.g. invalidateContent) we
		// allow for multiple views track a given file. In this case, we should
		// either only report diagnostics for snapshots from the "best" view of a
		// URI, or somehow merge diagnostics from multiple views.
		if r.publishedSnapshotID > snapshot.GlobalID() {
			continue
		}

		anyReportsChanged := false
		reportHashes := map[diagnosticSource]string{}
		var diags []*source.Diagnostic
		for dsource, report := range r.reports {
			if report.snapshotID != snapshot.GlobalID() {
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
		if hash == r.publishedHash && !r.mustPublish {
			// Update snapshotID to be the latest snapshot for which this diagnostic
			// hash is valid.
			r.publishedSnapshotID = snapshot.GlobalID()
			continue
		}
		var version int32
		if fh := snapshot.FindFile(uri); fh != nil { // file may have been deleted
			version = fh.Version()
		}
		if err := s.client.PublishDiagnostics(ctx, &protocol.PublishDiagnosticsParams{
			Diagnostics: toProtocolDiagnostics(diags),
			URI:         protocol.URIFromSpanURI(uri),
			Version:     version,
		}); err == nil {
			r.publishedHash = hash
			r.mustPublish = false // diagnostics have been successfully published
			r.publishedSnapshotID = snapshot.GlobalID()
			for dsource, hash := range reportHashes {
				report := r.reports[dsource]
				report.publishedHash = hash
				r.reports[dsource] = report
			}
		} else {
			if ctx.Err() != nil {
				// Publish may have failed due to a cancelled context.
				log.Trace.Log(ctx, "publish cancelled")
				return
			}
			event.Error(ctx, "publishReports: failed to deliver diagnostic", err, tag.URI.Of(uri))
		}
	}
}

func toProtocolDiagnostics(diagnostics []*source.Diagnostic) []protocol.Diagnostic {
	reports := []protocol.Diagnostic{}
	for _, diag := range diagnostics {
		pdiag := protocol.Diagnostic{
			// diag.Message might start with \n or \t
			Message:            strings.TrimSpace(diag.Message),
			Range:              diag.Range,
			Severity:           diag.Severity,
			Source:             string(diag.Source),
			Tags:               diag.Tags,
			RelatedInformation: diag.Related,
		}
		if diag.Code != "" {
			pdiag.Code = diag.Code
		}
		if diag.CodeHref != "" {
			pdiag.CodeDescription = &protocol.CodeDescription{Href: diag.CodeHref}
		}
		reports = append(reports, pdiag)
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

// Diagnostics formattedfor the debug server
// (all the relevant fields of Server are private)
// (The alternative is to export them)
func (s *Server) Diagnostics() map[string][]string {
	ans := make(map[string][]string)
	s.diagnosticsMu.Lock()
	defer s.diagnosticsMu.Unlock()
	for k, v := range s.diagnostics {
		fn := k.Filename()
		for typ, d := range v.reports {
			if len(d.diags) == 0 {
				continue
			}
			for _, dx := range d.diags {
				ans[fn] = append(ans[fn], auxStr(dx, d, typ))
			}
		}
	}
	return ans
}

func auxStr(v *source.Diagnostic, d diagnosticReport, typ diagnosticSource) string {
	// Tags? RelatedInformation?
	msg := fmt.Sprintf("(%s)%q(source:%q,code:%q,severity:%s,snapshot:%d,type:%s)",
		v.Range, v.Message, v.Source, v.Code, v.Severity, d.snapshotID, typ)
	for _, r := range v.Related {
		msg += fmt.Sprintf(" [%s:%s,%q]", r.Location.URI.SpanURI().Filename(), r.Location.Range, r.Message)
	}
	return msg
}
