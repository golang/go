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
	"sort"
	"strings"
	"sync"
	"time"

	"golang.org/x/tools/gopls/internal/bug"
	"golang.org/x/tools/gopls/internal/lsp/mod"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/lsp/template"
	"golang.org/x/tools/gopls/internal/lsp/work"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/event/tag"
)

// TODO(rfindley): simplify this very complicated logic for publishing
// diagnostics. While doing so, ensure that we can test subtle logic such as
// for multi-pass diagnostics.

// diagnosticSource differentiates different sources of diagnostics.
//
// Diagnostics from the same source overwrite each other, whereas diagnostics
// from different sources do not. Conceptually, the server state is a mapping
// from diagnostics source to a set of diagnostics, and each storeDiagnostics
// operation updates one entry of that mapping.
type diagnosticSource int

const (
	modParseSource diagnosticSource = iota
	modTidySource
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
	reports map[diagnosticSource]*diagnosticReport
}

func (d diagnosticSource) String() string {
	switch d {
	case modParseSource:
		return "FromModParse"
	case modTidySource:
		return "FromModTidy"
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
//
// hashDiagnostics mutates its argument (via sorting).
func hashDiagnostics(diags ...*source.Diagnostic) string {
	if len(diags) == 0 {
		return emptyDiagnosticsHash
	}
	return computeDiagnosticHash(diags...)
}

// opt: pre-computed hash for empty diagnostics
var emptyDiagnosticsHash = computeDiagnosticHash()

// computeDiagnosticHash should only be called from hashDiagnostics.
//
// TODO(rfindley): this should use source.Hash.
func computeDiagnosticHash(diags ...*source.Diagnostic) string {
	source.SortDiagnostics(diags)
	h := sha256.New()
	for _, d := range diags {
		for _, t := range d.Tags {
			fmt.Fprintf(h, "tag: %s\n", t)
		}
		for _, r := range d.Related {
			fmt.Fprintf(h, "related: %s %s %s\n", r.Location.URI.SpanURI(), r.Message, r.Location.Range)
		}
		fmt.Fprintf(h, "message: %s\n", d.Message)
		fmt.Fprintf(h, "range: %s\n", d.Range)
		fmt.Fprintf(h, "severity: %s\n", d.Severity)
		fmt.Fprintf(h, "source: %s\n", d.Source)
		if d.BundledFixes != nil {
			fmt.Fprintf(h, "fixes: %s\n", *d.BundledFixes)
		}
	}
	return fmt.Sprintf("%x", h.Sum(nil))
}

func (s *Server) diagnoseSnapshots(snapshots map[source.Snapshot][]span.URI, onDisk bool) {
	var diagnosticWG sync.WaitGroup
	for snapshot, uris := range snapshots {
		diagnosticWG.Add(1)
		go func(snapshot source.Snapshot, uris []span.URI) {
			defer diagnosticWG.Done()
			s.diagnoseSnapshot(snapshot, uris, onDisk, snapshot.View().Options().DiagnosticsDelay)
		}(snapshot, uris)
	}
	diagnosticWG.Wait()
}

// diagnoseSnapshot computes and publishes diagnostics for the given snapshot.
//
// If delay is non-zero, computing diagnostics does not start until after this
// delay has expired, to allow work to be cancelled by subsequent changes.
//
// If changedURIs is non-empty, it is a set of recently changed files that
// should be diagnosed immediately, and onDisk reports whether these file
// changes came from a change to on-disk files.
func (s *Server) diagnoseSnapshot(snapshot source.Snapshot, changedURIs []span.URI, onDisk bool, delay time.Duration) {
	ctx := snapshot.BackgroundContext()
	ctx, done := event.Start(ctx, "Server.diagnoseSnapshot", source.SnapshotLabels(snapshot)...)
	defer done()

	if delay > 0 {
		// 2-phase diagnostics.
		//
		// The first phase just parses and type-checks (but
		// does not analyze) packages directly affected by
		// file modifications.
		//
		// The second phase runs after the delay, and does everything.
		s.diagnoseChangedFiles(ctx, snapshot, changedURIs, onDisk)
		s.publishDiagnostics(ctx, false, snapshot)

		select {
		case <-time.After(delay):
		case <-ctx.Done():
			return
		}
	}

	s.diagnose(ctx, snapshot, analyzeOpenPackages)
	s.publishDiagnostics(ctx, true, snapshot)
}

func (s *Server) diagnoseChangedFiles(ctx context.Context, snapshot source.Snapshot, uris []span.URI, onDisk bool) {
	ctx, done := event.Start(ctx, "Server.diagnoseChangedFiles", source.SnapshotLabels(snapshot)...)
	defer done()

	toDiagnose := make(map[source.PackageID]*source.Metadata)
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
		if snapshot.IsBuiltin(uri) {
			continue
		}

		// Don't diagnose files that are ignored by `go list` (e.g. testdata).
		if snapshot.IgnoredFile(uri) {
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
		source.RemoveIntermediateTestVariants(&metas)
		for _, m := range metas {
			toDiagnose[m.ID] = m
		}
	}
	s.diagnosePkgs(ctx, snapshot, toDiagnose, nil)
}

// analysisMode parameterizes analysis behavior of a call to diagnosePkgs.
type analysisMode int

const (
	analyzeNothing      analysisMode = iota // don't run any analysis
	analyzeOpenPackages                     // run analysis on packages with open files
	analyzeEverything                       // run analysis on all packages
)

// diagnose is a helper function for running diagnostics with a given context.
// Do not call it directly. forceAnalysis is only true for testing purposes.
func (s *Server) diagnose(ctx context.Context, snapshot source.Snapshot, analyze analysisMode) {
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

	// Diagnostics below are organized by increasing specificity:
	//  go.work > mod > mod upgrade > mod vuln > package, etc.

	// Diagnose go.work file.
	workReports, workErr := work.Diagnostics(ctx, snapshot)
	if ctx.Err() != nil {
		return
	}
	store(workSource, "diagnosing go.work file", workReports, workErr, true)

	// Diagnose go.mod file.
	modReports, modErr := mod.Diagnostics(ctx, snapshot)
	if ctx.Err() != nil {
		return
	}
	store(modParseSource, "diagnosing go.mod file", modReports, modErr, true)

	// Diagnose go.mod upgrades.
	upgradeReports, upgradeErr := mod.UpgradeDiagnostics(ctx, snapshot)
	if ctx.Err() != nil {
		return
	}
	store(modCheckUpgradesSource, "diagnosing go.mod upgrades", upgradeReports, upgradeErr, true)

	// Diagnose vulnerabilities.
	vulnReports, vulnErr := mod.VulnerabilityDiagnostics(ctx, snapshot)
	if ctx.Err() != nil {
		return
	}
	store(modVulncheckSource, "diagnosing vulnerabilities", vulnReports, vulnErr, false)

	workspace, err := snapshot.WorkspaceMetadata(ctx)
	if s.shouldIgnoreError(ctx, snapshot, err) {
		return
	}
	criticalErr := snapshot.CriticalError(ctx)
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
	if len(workspace) == 0 {
		return
	}

	var wg sync.WaitGroup // for potentially slow operations below

	// Maybe run go mod tidy (if it has been invalidated).
	//
	// Since go mod tidy can be slow, we run it concurrently to diagnostics.
	wg.Add(1)
	go func() {
		defer wg.Done()
		modTidyReports, err := mod.TidyDiagnostics(ctx, snapshot)
		store(modTidySource, "running go mod tidy", modTidyReports, err, true)
	}()

	// Run type checking and go/analysis diagnosis of packages in parallel.
	var (
		seen       = map[span.URI]struct{}{}
		toDiagnose = make(map[source.PackageID]*source.Metadata)
		toAnalyze  = make(map[source.PackageID]unit)
	)
	for _, m := range workspace {
		var hasNonIgnored, hasOpenFile bool
		for _, uri := range m.CompiledGoFiles {
			seen[uri] = struct{}{}
			if !hasNonIgnored && !snapshot.IgnoredFile(uri) {
				hasNonIgnored = true
			}
			if !hasOpenFile && snapshot.IsOpen(uri) {
				hasOpenFile = true
			}
		}
		if hasNonIgnored {
			toDiagnose[m.ID] = m
			if analyze == analyzeEverything || analyze == analyzeOpenPackages && hasOpenFile {
				toAnalyze[m.ID] = unit{}
			}
		}
	}

	wg.Add(1)
	go func() {
		s.diagnosePkgs(ctx, snapshot, toDiagnose, toAnalyze)
		wg.Done()
	}()

	wg.Wait()

	// Orphaned files.
	// Confirm that every opened file belongs to a package (if any exist in
	// the workspace). Otherwise, add a diagnostic to the file.
	if diags, err := snapshot.OrphanedFileDiagnostics(ctx); err == nil {
		for uri, diag := range diags {
			s.storeDiagnostics(snapshot, uri, orphanedSource, []*source.Diagnostic{diag}, true)
		}
	} else {
		if ctx.Err() == nil {
			event.Error(ctx, "computing orphaned file diagnostics", err, source.SnapshotLabels(snapshot)...)
		}
	}
}

// diagnosePkgs type checks packages in toDiagnose, and analyzes packages in
// toAnalyze, merging their diagnostics. Packages in toAnalyze must be a subset
// of the packages in toDiagnose.
//
// It also implements gc_details diagnostics.
//
// TODO(rfindley): revisit handling of analysis gc_details. It may be possible
// to merge this function with Server.diagnose, thereby avoiding the two layers
// of concurrent dispatch: as of writing we concurrently run TidyDiagnostics
// and diagnosePkgs, and diagnosePkgs concurrently runs PackageDiagnostics and
// analysis.
func (s *Server) diagnosePkgs(ctx context.Context, snapshot source.Snapshot, toDiagnose map[source.PackageID]*source.Metadata, toAnalyze map[source.PackageID]unit) {
	ctx, done := event.Start(ctx, "Server.diagnosePkgs", source.SnapshotLabels(snapshot)...)
	defer done()

	// Analyze and type-check concurrently, since they are independent
	// operations.
	var (
		wg            sync.WaitGroup
		pkgDiags      map[span.URI][]*source.Diagnostic
		analysisDiags = make(map[span.URI][]*source.Diagnostic)
	)

	// Collect package diagnostics.
	wg.Add(1)
	go func() {
		defer wg.Done()
		var ids []source.PackageID
		for id := range toDiagnose {
			ids = append(ids, id)
		}
		var err error
		pkgDiags, err = snapshot.PackageDiagnostics(ctx, ids...)
		if err != nil {
			event.Error(ctx, "warning: diagnostics failed", err, source.SnapshotLabels(snapshot)...)
		}
	}()

	// Get diagnostics from analysis framework.
	// This includes type-error analyzers, which suggest fixes to compiler errors.
	wg.Add(1)
	go func() {
		defer wg.Done()
		diags, err := source.Analyze(ctx, snapshot, toAnalyze, false)
		if err != nil {
			var tagStr string // sorted comma-separated list of package IDs
			{
				// TODO(adonovan): replace with a generic map[S]any -> string
				// function in the tag package, and use  maps.Keys + slices.Sort.
				keys := make([]string, 0, len(toDiagnose))
				for id := range toDiagnose {
					keys = append(keys, string(id))
				}
				sort.Strings(keys)
				tagStr = strings.Join(keys, ",")
			}
			event.Error(ctx, "warning: analyzing package", err, append(source.SnapshotLabels(snapshot), tag.Package.Of(tagStr))...)
			return
		}
		for uri, diags := range diags {
			analysisDiags[uri] = append(analysisDiags[uri], diags...)
		}
	}()

	wg.Wait()

	// TODO(rfindley): remove the guards against snapshot.IsBuiltin, after the
	// gopls@v0.12.0 release. Packages should not be producing diagnostics for
	// the builtin file: I do not know why this logic existed previously.

	// Merge analysis diagnostics with package diagnostics, and store the
	// resulting analysis diagnostics.
	for uri, adiags := range analysisDiags {
		if snapshot.IsBuiltin(uri) {
			bug.Reportf("go/analysis reported diagnostics for the builtin file: %v", adiags)
			continue
		}
		tdiags := pkgDiags[uri]
		var tdiags2, adiags2 []*source.Diagnostic
		source.CombineDiagnostics(tdiags, adiags, &tdiags2, &adiags2)
		pkgDiags[uri] = tdiags2
		s.storeDiagnostics(snapshot, uri, analysisSource, adiags2, true)
	}

	// golang/go#59587: guarantee that we store type-checking diagnostics for every compiled
	// package file.
	//
	// Without explicitly storing empty diagnostics, the eager diagnostics
	// publication for changed files will not publish anything for files with
	// empty diagnostics.
	storedPkgDiags := make(map[span.URI]bool)
	for _, m := range toDiagnose {
		for _, uri := range m.CompiledGoFiles {
			s.storeDiagnostics(snapshot, uri, typeCheckSource, pkgDiags[uri], true)
			storedPkgDiags[uri] = true
		}
	}
	// Store the package diagnostics.
	for uri, diags := range pkgDiags {
		if storedPkgDiags[uri] {
			continue
		}
		// builtin.go exists only for documentation purposes, and is not valid Go code.
		// Don't report distracting errors
		if snapshot.IsBuiltin(uri) {
			bug.Reportf("type checking reported diagnostics for the builtin file: %v", diags)
			continue
		}
		s.storeDiagnostics(snapshot, uri, typeCheckSource, diags, true)
	}

	// Process requested gc_details diagnostics.
	//
	// TODO(rfindley): this could be improved:
	//   1. This should memoize its results if the package has not changed.
	//   2. This should not even run gc_details if the package contains unsaved
	//      files.
	//   3. See note below about using FindFile.
	var toGCDetail map[source.PackageID]*source.Metadata
	s.gcOptimizationDetailsMu.Lock()
	for id := range s.gcOptimizationDetails {
		if m, ok := toDiagnose[id]; ok {
			if toGCDetail == nil {
				toGCDetail = make(map[source.PackageID]*source.Metadata)
			}
			toGCDetail[id] = m
		}
	}
	s.gcOptimizationDetailsMu.Unlock()

	for _, m := range toGCDetail {
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
				// TODO(rfindley): remove the use of FindFile here, and use ReadFile
				// instead. Isn't it enough to know that the package came from the
				// snapshot? Any reports should apply to the snapshot.
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
			reports:       map[diagnosticSource]*diagnosticReport{},
		}
	}
	s.diagnostics[uri].mustPublish = true
}

// storeDiagnostics stores results from a single diagnostic source. If merge is
// true, it merges results into any existing results for this snapshot.
//
// Mutates (sorts) diags.
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
			reports:       map[diagnosticSource]*diagnosticReport{},
		}
	}
	report := s.diagnostics[uri].reports[dsource]
	if report == nil {
		report = new(diagnosticReport)
		s.diagnostics[uri].reports[dsource] = report
	}
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
			s.storeDiagnostics(snapshot, d.URI, modParseSource, []*source.Diagnostic{d}, true)
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
			// When we publish diagnostics for a file, we must update the
			// publishedHash for every report, not just the reports that were
			// published. Eliding a report is equivalent to publishing empty
			// diagnostics.
			for dsource, report := range r.reports {
				if hash, ok := reportHashes[dsource]; ok {
					report.publishedHash = hash
				} else {
					// The report was not (yet) stored for this snapshot. Record that we
					// published no diagnostics from this source.
					report.publishedHash = hashDiagnostics()
				}
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
		pdiag := protocol.Diagnostic{
			// diag.Message might start with \n or \t
			Message:            strings.TrimSpace(diag.Message),
			Range:              diag.Range,
			Severity:           diag.Severity,
			Source:             string(diag.Source),
			Tags:               emptySliceDiagnosticTag(diag.Tags),
			RelatedInformation: diag.Related,
			Data:               diag.BundledFixes,
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
	// TODO(rfindley): surely it is not correct to walk the folder here just to
	// suppress diagnostics, every time we compute diagnostics.
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

func auxStr(v *source.Diagnostic, d *diagnosticReport, typ diagnosticSource) string {
	// Tags? RelatedInformation?
	msg := fmt.Sprintf("(%s)%q(source:%q,code:%q,severity:%s,snapshot:%d,type:%s)",
		v.Range, v.Message, v.Source, v.Code, v.Severity, d.snapshotID, typ)
	for _, r := range v.Related {
		msg += fmt.Sprintf(" [%s:%s,%q]", r.Location.URI.SpanURI().Filename(), r.Location.Range, r.Message)
	}
	return msg
}
