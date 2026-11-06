// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

// This file implements the build actions for analysis in go test
// (see load/analysis.go).
//
// There is one analysis tool per go test invocation, containing all
// variables (analyzers and suites) referenced by the configurations
// that enable analyzers for any root package. (A root package is one
// for which go test reports diagnostics, as opposed to a dependency,
// which is analyzed only for the facts its importers need.) The
// tool's driver package is the standard library's
// internal/analysis/gotestchecker.
//
// The action graph is planned before the tool exists, so its topology
// is conservative: one analysisvet action per package in the import
// closure of the root packages, isomorphic to the import graph,
// plus an analysismeta action that runs the tool's metadata query
// (-describe) once the tool is linked, and one analysisreport action
// per root package:
//
//	link(tool) --> analysismeta --> analysisvet(P) --> analysisreport(P) --> run(P.test)
//	                                      ^
//	                                      |
//	                                analysisvet(Q), for each import Q of P
//
// Every analysisvet action depends on the analysismeta action, which
// performs the dynamic stage of resolution: from the metadata (the
// members of each suite, the Requires graph, and which analyzers use
// facts) it expands suites, applies selectors and negations, and
// computes, for each root package, the set of analyzers whose
// diagnostics are reported, and for each analysisvet action, the set
// of analyzers whose results it must produce. An analysisvet action whose set is
// empty merely produces types.
//
// Within the tool, analyzers are identified by pointer; in the
// protocol, they are identified by their index in the tool's
// metadata, a topological order that is deterministic for a given
// tool. Across tools, an analyzer is identified (for caching) by its
// Name and Run function and the variables from which it is
// reachable; see [Builder.analyzerIDs].
//
// Each analysisvet action produces, for each needed analyzer, one
// pass file containing that analyzer's facts (if it uses facts) and,
// for the analyzers reported for a root package, its diagnostics.
// So the facts of an analyzer required by several roots are computed
// and stored once per package. A pass file may instead record a
// failure, which prevents dependent passes and is reported at each
// affected root package.
//
// Analysis results are cached not per tool but per (package,
// analyzer), so that results are shared across invocations even
// though their tools differ:
//
//   - the types of package P are keyed by the inputs of P, including
//     the types of its direct imports and the target configuration,
//     and the dependency identity of the driver package;
//   - the pass file of analyzer A on P is keyed by the inputs of P,
//     the identity of A, whether A's diagnostics are reported,
//     and the facts IDs of the pass files, for each direct import of P,
//     of each analyzer that uses facts in A's Requires closure.
//     (The facts ID of a pass file excludes its diagnostics, which
//     depend on whether they are reported, so that the pass files of
//     a package are shared by invocations that do and do not vet
//     its dependencies.)
//
// The dependency identity of a package is a Merkle hash over its
// import graph that covers the content IDs of the compiled packages
// in its import closure. The identity of an analyzer is the hash of
// its Name and Run function (by run-time symbol name), the dependency
// identity of the driver package, and the dependency identities of
// the packages that declare the linked variables from which the
// analyzer is reachable. It does not depend on which other analyzers
// are linked into the tool.
// Analyzers linked into the same tool are assumed not to interfere
// with one another.

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"internal/analysis/gotestchecker/protocol"
	"maps"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"

	"cmd/go/internal/cache"
	"cmd/go/internal/cfg"
	"cmd/go/internal/load"
	"cmd/go/internal/modload"
	"cmd/go/internal/str"
)

// analysisResult records the outputs of an analysisvet action.
// It is the action's Provider.
type analysisResult struct {
	types    analysisOutput         // the package's types
	typeErrs []string               // type errors of the package (reported once)
	passes   map[int]analysisOutput // analyzer index -> pass file
}

// An analysisOutput is a file in the cache and its output ID, plus,
// for a pass file, information read from it (see readPassInfo).
type analysisOutput struct {
	file    string
	id      cache.OutputID
	factsID cache.OutputID    // for pass files of analyzers that use facts
	trivial bool              // pass file has no facts, no failure, and was not skipped
	failure *protocol.Failure // failure of the pass
}

// analysisResultOf returns the result of analysisvet action a, or nil
// if the action has not produced one (because it failed, or did not
// run).
func analysisResultOf(a *Action) *analysisResult {
	res, _ := a.Provider.(*analysisResult)
	return res
}

// The types of the protocol between the go command and the analysis
// tool are defined by internal/analysis/gotestchecker/protocol.

// analysisState is the per-Builder state of analysis in go test.
type analysisState struct {
	actions    []*Action                        // all actions of Mode "analysisvet"
	roots      []*Action                        // analysisvet actions of root packages, in order of request
	rootConfig map[*Action]*load.AnalysisConfig // root analysisvet action -> references that apply to its package
	tool       *Action                          // link action for the tool
	meta       *Action                          // analysismeta action

	// toolPath is the executable of the tool. toolID identifies
	// the executable for caching its metadata ("" if not cacheable).
	toolPath, toolID string

	// plan is set by the analysismeta action, before any
	// analysisvet action runs, if the metadata is available.
	plan *analysisPlan
}

// An analysisPlan holds the metadata of the analysis tool's analyzers
// (by index) and the plan derived from it.
type analysisPlan struct {
	analyzers []analyzerInfo                        // index -> analyzer
	driverID  cache.ActionID                        // dependency identity of the driver package
	need      map[*Action][]int                     // analysisvet action -> sorted indexes of needed pass files
	reported  map[*Action]map[int]*reportedAnalyzer // root analysisvet action -> index -> analyzer whose diagnostics are reported
	refErrs   map[*Action][]string                  // root analysisvet action -> errors in its references
}

// An analyzerInfo holds the metadata of one of the analysis tool's
// analyzers and information derived from it.
type analyzerInfo struct {
	*protocol.Analyzer                // metadata (Name, Requires, Facts, etc.)
	reqClosure         []int          // sorted indexes of Requires closure (inclusive)
	id                 cache.ActionID // identity
}

// A reportedAnalyzer describes an analyzer whose diagnostics are
// reported for a root package. (The pass of that analyzer on that
// package is a reported pass.)
type reportedAnalyzer struct {
	name string // "pkg.Var" or "pkg.Var[Name]"

	// TODO(adonovan): more to come
}

// AnalysisVetAction returns an action that reports the diagnostics of
// the analyzers enabled for package p by the specified configuration,
// which must be non-nil. The action depends on the analysis of p,
// which in turn depends on the analysis of each of p's dependencies,
// which computes the facts needed by p's analyzers.
// The tool must be supplied later by SetAnalysisTool.
//
// Reporting is a separate action from analysis because the analysis
// action of a root package may be shared with the importers of that
// package (e.g. a package without test files is itself the root, not
// a test variant of it), and its diagnostics must not cause the
// analysis of its importers to fail.
func (b *Builder) AnalysisVetAction(s *modload.Loader, p *load.Package, config *load.AnalysisConfig) *Action {
	if config == nil {
		panic("nil config")
	}
	a := b.analysisVetAction(s, p)
	st := &b.analysis
	if _, ok := st.rootConfig[a]; !ok { // first request to vet p
		st.roots = append(st.roots, a)
		if st.rootConfig == nil {
			st.rootConfig = make(map[*Action]*load.AnalysisConfig)
		}
	}
	st.rootConfig[a] = config // (makes p a root package)
	return &Action{
		Mode:       "analysisreport",
		Package:    p,
		Deps:       []*Action{a},
		Actor:      ActorFunc((*Builder).analysisReport),
		IgnoreFail: true, // report diagnostics even if some analyzers failed
	}
}

// analysisVetAction returns the analysisvet action for package p,
// creating it (and those of p's dependencies) if necessary. The action
// depends on the build of p and on the analysisvet action of each of
// p's imports, from which it obtains facts.
func (b *Builder) analysisVetAction(s *modload.Loader, p *load.Package) *Action {
	return b.cacheAction("analysisvet", p, func() *Action {
		a1 := b.BuildExportAction(ModeBuild, ModeBuild, p)
		deps := []*Action{a1}
		for _, p1 := range p.Internal.Imports {
			deps = append(deps, b.analysisVetAction(s, p1))
		}
		a := &Action{
			Mode:       "analysisvet",
			Package:    p,
			Deps:       deps,
			Objdir:     b.NewObjdir(),
			VetxOnly:   true,
			IgnoreFail: true, // run the passes not downstream of a failed dependency (see analysisVet)
		}
		b.analysis.actions = append(b.analysis.actions, a)
		if a1.Actor == nil {
			// built-in package, such as unsafe
			return a
		}
		if p.UsesCgo() || p.UsesSwig() {
			a.Deps = append(a.Deps, b.runCgoAction(p, false))
		}
		a.Actor = ActorFunc((*Builder).analysisVet)
		b.addTransitiveCompileActions(a, a1)
		return a
	})
}

// SetAnalysisTool records the link action of the analysis tool. It
// creates the analysismeta action, which depends on the link action,
// and makes every analysis action depend on it.
func (b *Builder) SetAnalysisTool(link *Action) {
	st := &b.analysis
	st.tool = link
	b.setAnalysisMeta(&Action{
		Mode:   "analysismeta",
		Deps:   []*Action{link},
		Objdir: b.NewObjdir(),
		Actor:  ActorFunc((*Builder).analysisMeta),
	})
}

// setAnalysisMeta records the analysismeta action and makes every
// analysis action depend on it.
func (b *Builder) setAnalysisMeta(meta *Action) {
	st := &b.analysis
	st.meta = meta
	for _, a := range st.actions {
		if a.Actor != nil {
			a.Deps = append(a.Deps, meta)
		}
	}
}

// prioritizeAnalysisTool gives the actions that build the analysis
// tool and query its metadata precedence over all others, preserving
// their relative order. No analysis can start until the metadata is
// available, so building the tool early lets analysis proceed as soon
// as each package is compiled, interleaved with the rest of the build,
// rather than in a tail at the end. (The tool's dependencies include
// std packages that the build needs anyway.)
func (b *Builder) prioritizeAnalysisTool(all []*Action) {
	if meta := b.analysis.meta; meta != nil {
		for _, a := range actionList(meta) {
			// Deps gate readiness; priority orders ready actions.
			// This is the only adjustment to Do's DFS priorities.
			a.priority -= len(all)
		}
	}
}

// HasAnalysisActions reports whether any analysis actions were created.
func (b *Builder) HasAnalysisActions() bool { return len(b.analysis.actions) > 0 }

// analysisMeta is the actor for the analysismeta action.
// It runs the tool's metadata query and plans the analysis.
func (b *Builder) analysisMeta(ctx context.Context, a *Action) error {
	st := &b.analysis
	st.toolPath = st.tool.BuiltTarget()
	st.toolID = st.tool.buildID
	data, err := b.analysisMetadata(a)
	if err != nil || data == nil {
		return err // (data is nil with -n)
	}
	md := new(protocol.Metadata)
	if err := json.Unmarshal(data, md); err != nil {
		return fmt.Errorf("decoding analysis tool metadata: %v", err)
	}
	driverID, ids := b.analyzerIDs(md)
	b.planAnalysis(md, driverID, ids)
	return nil
}

// analysisMetadata returns the result of the tool's metadata query,
// running the tool if necessary. The metadata is cached, keyed by
// the tool's build ID, so a warm run need not execute the tool.
// It returns nil with -n.
func (b *Builder) analysisMetadata(a *Action) ([]byte, error) {
	st := &b.analysis
	c := cache.Default()
	h := cache.NewHash("analysis metadata")
	fmt.Fprintf(h, "tool %s\n", st.toolID)
	key := h.Sum()

	if !cfg.BuildA && st.toolID != "" {
		if data, _, err := cache.GetBytes(c, key); err == nil {
			return data, nil
		}
	}
	sh := b.Shell(a)
	if err := sh.Mkdir(a.Objdir); err != nil {
		return nil, err
	}
	file := a.Objdir + "metadata.json"
	if err := sh.run(a.Objdir, "analysis tool", nil, cfg.BuildToolexec, st.toolPath, "-describe="+file); err != nil {
		if cfg.Goos != runtime.GOOS || cfg.Goarch != runtime.GOARCH {
			// The tool is built for the tests' platform and run
			// on the host, which may not support it.
			return nil, fmt.Errorf("running analysis tool built for %s/%s: %v\n(to run the tests on another platform, use -exec; to disable analysis, use -vet=off)", cfg.Goos, cfg.Goarch, err)
		}
		return nil, err
	}
	if cfg.BuildN {
		return nil, nil
	}
	data, err := os.ReadFile(file)
	if err != nil {
		return nil, err
	}
	if st.toolID != "" {
		if err := cache.PutBytes(c, key, data); err != nil {
			return nil, err
		}
	}
	return data, nil
}

// analyzerIDs computes the dependency identity of the driver package
// and the identity of each analyzer of the metadata from the compiled
// packages linked into the tool.
//
// The identity of an analyzer derives from its Name and Run function
// (by run-time symbol name), and from the dependency identities of
// the packages that declare the linked variables from which it is
// reachable, as a member or by Requires edges. Their import closures
// contain all the code of the analyzer, wherever it is declared.
// The identity depends on which linked variables reach the analyzer,
// but not on what else is linked into the tool, so that results are
// shared by tools that link different sets of variables, and by all
// the roots that require an analyzer. (An edit to any member of a
// suite changes the identities of all its members, but such edits
// are rare except for the author of the suite.)
//
// Distinct analyzers with the same Name and Run function ("twins")
// reachable from the same variables have the same identity. The tool
// rejects twins within the Requires closure of one member; twins in
// different closures are not detected (unsound, but they do not arise
// by accident).
func (b *Builder) analyzerIDs(md *protocol.Metadata) (driverID cache.ActionID, ids []cache.ActionID) {
	st := &b.analysis
	compiled := make(map[string]*Action) // import path -> compile action
	for _, a1 := range st.tool.Deps {
		if a1.Package != nil {
			compiled[a1.Package.ImportPath] = a1
		}
	}
	// merkle returns the dependency identity of package p: a
	// Merkle hash over its import graph, combining the content ID
	// of p's compiled package with the dependency identities of
	// its direct imports. It thus covers the content IDs of all
	// the compiled packages in the import closure of p. It is
	// memoized per package, so the closures of different
	// packages, which overlap heavily, are each visited once.
	depsMemo := make(map[*load.Package]cache.ActionID)
	var merkle func(p *load.Package) cache.ActionID
	merkle = func(p *load.Package) cache.ActionID {
		if id, ok := depsMemo[p]; ok {
			return id
		}
		h := cache.NewHash("analysis deps")
		fmt.Fprintf(h, "package %q\n", p.ImportPath)
		// Built-in packages such as unsafe
		// have no compile action.
		if a1 := compiled[p.ImportPath]; a1 != nil {
			fmt.Fprintf(h, "content %s\n", a1.BuildContentID())
		}
		// Combine the identities of the direct imports by
		// addition (mod 2^256), which is independent of their
		// order (p.Internal.Imports is not necessarily sorted).
		// XOR would also be, but a test variant's Imports may
		// list a package twice (once from the test files), and
		// XOR would cancel the pair.
		var imports cache.ActionID
		for _, imp := range p.Internal.Imports {
			addActionID(&imports, merkle(imp))
		}
		fmt.Fprintf(h, "imports %x\n", imports)
		id := h.Sum()
		depsMemo[p] = id
		return id
	}
	// The driver package is always linked into the tool, since the
	// synthesized main package imports it.
	driverID = merkle(compiled[load.AnalysisToolDriver].Package)

	// varPkgs[h] is the set of packages that declare a linked
	// variable from whose members analyzer h is reachable by
	// Requires edges.
	//
	// We assume that a variable's value, and everything reachable
	// from it (its analyzers, their prerequisites, their Run
	// functions, and the code those call), is determined by code
	// within the import closure of the variable's package, so that
	// the closures of these packages contain all the code of the
	// analyzer, wherever it is declared. The assumption fails only
	// if code outside the closure modifies the variable (for
	// example, a checker package that registers itself from an
	// init function in a registry variable that it imports). Such
	// code is linked into the tool only if the package of another
	// variable imports it, so a failure implies interference
	// between variables, which we already exclude (see the package
	// comment). See also the addendum on analyzer identity in the
	// design doc for go.dev/issue/82464.
	varPkgs := make([]map[string]bool, len(md.Analyzers))
	for _, v := range md.Vars {
		if v.Error != "" {
			continue // Members undefined
		}
		pkg := v.Symbol[:strings.LastIndexByte(v.Symbol, '.')]
		var visit func(h int)
		visit = func(h int) {
			if varPkgs[h] == nil {
				varPkgs[h] = make(map[string]bool)
			}
			if !varPkgs[h][pkg] {
				varPkgs[h][pkg] = true
				for _, req := range md.Analyzers[h].Requires {
					visit(req)
				}
			}
		}
		for _, m := range v.Members {
			if m.Analyzer >= 0 {
				visit(m.Analyzer)
			}
		}
	}

	ids = make([]cache.ActionID, len(md.Analyzers))
	for h, an := range md.Analyzers {
		hash := cache.NewHash("analyzer")
		fmt.Fprintf(hash, "name %q\nrun %q\ndriver %x\n", an.Name, an.RunFunc, driverID)
		// Combine the dependency identities of the packages
		// of varPkgs[h] by addition (see merkle).
		var vars cache.ActionID
		linked := len(varPkgs[h]) > 0
		for pkg := range varPkgs[h] {
			if a1 := compiled[pkg]; a1 != nil {
				addActionID(&vars, merkle(a1.Package))
			} else {
				linked = false
			}
		}
		fmt.Fprintf(hash, "vars %x\n", vars)
		if !linked {
			// The packages of the variables are not
			// known. Use the identity of the tool, which
			// is sound but defeats sharing.
			fmt.Fprintf(hash, "tool %s\n", st.tool.buildID)
		}
		ids[h] = hash.Sum()
	}
	return driverID, ids
}

// addActionID sets *sum to *sum + id, treating both as 256-bit
// little-endian integers and discarding the carry out.
func addActionID(sum *cache.ActionID, id cache.ActionID) {
	var carry uint16
	for i := range sum {
		s := uint16(sum[i]) + uint16(id[i]) + carry
		sum[i] = byte(s)
		carry = s >> 8
	}
}

// planAnalysis performs the dynamic stage of resolution, computing
// the analyzers reported for each root package and the set of pass
// files needed from each analysisvet action. It sets st.plan.
// driverID and ids are the identities of the driver package and of each
// analyzer (see analyzerIDs).
func (b *Builder) planAnalysis(md *protocol.Metadata, driverID cache.ActionID, ids []cache.ActionID) {
	st := &b.analysis

	// Index the metadata. Analyzers are identified by their index
	// in md.Analyzers, which is in topological order (prerequisites
	// first).
	plan := &analysisPlan{
		analyzers: make([]analyzerInfo, len(md.Analyzers)),
		driverID:  driverID,
		reported:  make(map[*Action]map[int]*reportedAnalyzer),
		refErrs:   make(map[*Action][]string),
	}
	for i, an := range md.Analyzers {
		set := map[int]bool{i: true}
		for _, req := range an.Requires {
			for _, j := range plan.analyzers[req].reqClosure { // (computed already)
				set[j] = true
			}
		}
		plan.analyzers[i] = analyzerInfo{
			Analyzer:   an,
			reqClosure: slices.Sorted(maps.Keys(set)),
			id:         ids[i],
		}
	}

	// closure returns the sorted union of the Requires closures of
	// the analyzers hs.
	closure := func(hs []int) []int {
		set := make(map[int]bool)
		for _, h := range hs {
			for _, h2 := range plan.analyzers[h].reqClosure {
				set[h2] = true
			}
		}
		return slices.Sorted(maps.Keys(set))
	}

	// Expand suites, apply selectors and negations, and compute
	// the analyzers reported for each root package.
	for _, a := range st.roots {
		plan.reported[a], plan.refErrs[a] = resolveAnalyzers(md, st.rootConfig[a])
	}

	// Compute the pass files needed from each analysisvet action:
	// one for each pass that runs, whether or not it has facts or
	// diagnostics, so that its outcome (including a failure) is
	// recorded in exactly one place. A root package runs the
	// Requires closure of its reported analyzers; it and each of
	// its transitive dependencies run the Requires closure of the
	// analyzers that use facts in that closure. (Most such pass
	// files are trivial, and are not read by importers.)
	need := make(map[*Action]map[int]bool)
	// add adds the analyzers hs to the needs of action a, and
	// reports whether any of them was new.
	add := func(a *Action, hs []int) bool {
		set := need[a]
		if set == nil {
			set = make(map[int]bool)
			need[a] = set
		}
		added := false
		for _, h := range hs {
			if !set[h] {
				set[h] = true
				added = true
			}
		}
		return added
	}
	// propagate adds the analyzers hs, a Requires closure of
	// analyzers that use facts, to the needs of each analysisvet
	// dependency of a, and recursively of theirs. It stops at a
	// dependency that already had them all: its own dependencies
	// were given them when it was.
	//
	// (It propagates to the dependencies directly: the needs of a
	// root package, the closure of its reported analyzers,
	// already include hs.)
	var propagate func(a *Action, hs []int)
	propagate = func(a *Action, hs []int) {
		for _, dep := range a.Deps {
			if dep.Mode == "analysisvet" && add(dep, hs) {
				propagate(dep, hs)
			}
		}
	}
	for _, a := range st.roots {
		run := closure(slices.Collect(maps.Keys(plan.reported[a])))
		add(a, run)
		var factUsers []int
		for _, h := range run {
			if plan.analyzers[h].Facts {
				factUsers = append(factUsers, h)
			}
		}
		propagate(a, closure(factUsers))
	}
	plan.need = make(map[*Action][]int)
	for a, set := range need {
		plan.need[a] = slices.Sorted(maps.Keys(set))
	}
	st.plan = plan
}

// resolveAnalyzers performs the dynamic stage of resolution for one
// root package, whose references are described by config. Using
// the tool's metadata, it expands suites, applies selectors, and
// applies negations by Analyzer.Name. It returns the analyzers whose diagnostics are reported, by index, and
// errors in the references.
//
// Validation is per reference: an invalid reference (or member) is
// reported and dropped; the others still run.
func resolveAnalyzers(md *protocol.Metadata, config *load.AnalysisConfig) (map[int]*reportedAnalyzer, []string) {
	var errs []string
	errorf := func(annot *load.AnalysisRef, format string, args ...any) {
		errs = append(errs, annot.Errorf(format, args...).Error())
	}

	// members returns the valid members selected by reference
	// annot, with their indexes and display names, reporting any
	// errors.
	type member struct {
		index int
		name  string
	}
	members := func(annot *load.AnalysisRef) []member {
		v := md.Var(annot.Var.String())
		if v == nil {
			return nil // not linked (error reported statically)
		}
		if v.Error != "" {
			errorf(annot, "%s", v.Error)
			return nil
		}
		if v.IsSuite != annot.IsSuite() {
			if !v.IsSuite {
				errorf(annot, "%s is an analyzer, not a suite (remove the brackets)", annot.Var)
			} else {
				errorf(annot, "%[1]s is a suite (use %[1]s[*] or select members by name)", annot.Var)
			}
			return nil
		}
		var res []member
		addMember := func(m *protocol.Member) {
			name := annot.Var.String()
			if v.IsSuite {
				name += "[" + m.Name + "]"
			}
			if m.Analyzer < 0 {
				msg := m.Error
				if v.IsSuite {
					msg = fmt.Sprintf("member %s: %s", m.Name, msg)
				}
				errorf(annot, "%s", msg)
				return
			}
			res = append(res, member{m.Analyzer, name})
		}
		if !annot.IsSuite() || annot.AllMembers() {
			for _, m := range v.Members {
				addMember(m)
			}
		} else {
			for _, name := range annot.Members {
				i := slices.IndexFunc(v.Members, func(m *protocol.Member) bool {
					return m.Name == name
				})
				if i < 0 {
					errorf(annot, "suite %s has no member named %s", annot.Var.String(), name)
					continue
				}
				addMember(v.Members[i])
			}
		}
		return res
	}

	// Enable the members of each positive reference. The first
	// (most specific) enabling reference of each analyzer wins.
	enabled := make(map[int]*reportedAnalyzer)
	for _, annot := range config.Positive {
		for _, m := range members(annot) {
			if _, ok := enabled[m.index]; !ok {
				enabled[m.index] = &reportedAnalyzer{name: m.name}
			}
		}
	}

	// Apply negations: a negation by name disables every enabled
	// analyzer of that name.
	for _, annot := range config.Negative {
		for h := range enabled {
			if md.Analyzers[h].Name == annot.Name {
				delete(enabled, h)
			}
		}
	}
	return enabled, errs
}

// analysisVet is the actor for an analysisvet action.
func (b *Builder) analysisVet(ctx context.Context, a *Action) error {
	// Clear the failure that the scheduler propagated from the
	// dependencies: this action decides for itself, pass by pass,
	// what it can do. The analysis of a dependency may have
	// failed (reported there), but its types are available, so
	// this package's types are computed, and the passes not
	// downstream of the failure run; those downstream are skipped.
	// Otherwise the outcome for this package would depend on which
	// other packages are roots (whose analyzers the dependency was
	// analyzed with).
	a.Failed = nil

	// a.Deps[0] is the build of the package being analyzed.
	if a.Deps[0].Failed != nil {
		return nil // the build of the package failed; skip analysis
	}
	for _, a1 := range a.Deps {
		if a1.Failed != nil && a1.Mode == "cgo run" {
			return nil // cgo failed; skip analysis
		}
	}
	st := &b.analysis
	if cfg.BuildN {
		// Without the tool's metadata, the analysis cannot be
		// planned, so show a placeholder command for each root
		// package.
		if _, ok := st.rootConfig[a]; ok {
			b.Shell(a).ShowCmd("", "%s %s # analyzers determined by -describe", st.toolPath, a.Objdir+"analysis.cfg")
		}
		return nil
	}
	plan := st.plan
	if plan == nil {
		return nil // tool or metadata query failed (reported elsewhere)
	}

	// Gather the results of the direct imports.
	var deps []*Action
	for _, a1 := range a.Deps {
		if a1.Mode == "analysisvet" && a1.Actor != nil {
			if analysisResultOf(a1) == nil {
				return nil // analysis of dependency failed (reported elsewhere)
			}
			deps = append(deps, a1)
		}
	}
	slices.SortFunc(deps, func(x, y *Action) int {
		return strings.Compare(x.Package.ImportPath, y.Package.ImportPath)
	})

	reported := func(index int) bool {
		_, ok := plan.reported[a][index]
		return ok
	}
	vcfg0, vetDir := buildVetConfig(a)

	// The vet config names files generated in vetDir (those generated
	// by cgo) relative to it, since cmd/vet runs there (see trimObjdir).
	// The tool runs in the package directory, so make them absolute.
	absFiles := func(files []string) []string {
		res := make([]string, len(files))
		for i, file := range files {
			if !filepath.IsAbs(file) {
				file = vetDir + file
			}
			res[i] = file
		}
		return res
	}
	var (
		goFiles      = absFiles(vcfg0.GoFiles)
		nonGoFiles   = absFiles(vcfg0.NonGoFiles)
		ignoredFiles = absFiles(vcfg0.IgnoredFiles)
	)

	// Compute a key for the package's inputs. (We don't use the
	// compile action ID, which depends on incidental details of the
	// action graph, such as whether a tested package is also
	// imported by another tested package.) The types of the direct
	// imports are represented by their output IDs, giving early
	// cutoff when a dependency changes but its API does not.
	pkgID := func() cache.ActionID {
		h := cache.NewHash("analysis pkg " + a.Package.ImportPath)
		fmt.Fprintf(h, "id %q\nimportpath %q\ngoversion %q\n", vcfg0.ID, vcfg0.ImportPath, vcfg0.GoVersion)
		// The target configuration.
		fmt.Fprintf(h, "goos %q\ngoarch %q\ntags %q\ntooltags %q\n", cfg.Goos, cfg.Goarch, cfg.BuildContext.BuildTags, cfg.BuildContext.ToolTags)
		if m := vcfg0.Module; m != nil {
			fmt.Fprintf(h, "module %q %q %q\n", m.Path, m.Version, m.GoVersion)
		}
		// Files are identified by name and content. The name
		// appears in positions, so it must be the actual name
		// (which may be that of an overlay replacement), except
		// for files generated (by cgo or -cover) in the work
		// directory, whose name varies from run to run, and which
		// are identified by base name.
		fmt.Fprintf(h, "dir %q\n", vcfg0.Dir)
		for _, files := range [][]string{goFiles, nonGoFiles, ignoredFiles} {
			for _, file := range files {
				name := file
				if str.HasFilePathPrefix(file, b.WorkDir) {
					name = filepath.Base(file)
				}
				fmt.Fprintf(h, "file %q %s\n", name, b.fileHash(file))
			}
			fmt.Fprintf(h, "\n")
		}
		// Hash only the import mappings that refer to actual
		// imports: the vet config may contain spurious ones
		// (e.g. a self-mapping added when a tested package is
		// also imported by another tested package).
		depPaths := map[string]bool{"unsafe": true}
		for _, a1 := range deps {
			depPaths[a1.Package.ImportPath] = true
		}
		for _, k := range slices.Sorted(maps.Keys(vcfg0.ImportMap)) {
			if v := vcfg0.ImportMap[k]; depPaths[v] {
				fmt.Fprintf(h, "importmap %q %q\n", k, v)
			}
		}
		for _, a1 := range deps {
			fmt.Fprintf(h, "dep %q %x\n", a1.Package.ImportPath, analysisResultOf(a1).types.id)
		}
		return h.Sum()
	}()

	// Compute the cache keys.
	typesKey := func() cache.ActionID {
		h := cache.NewHash("analysis types " + a.Package.ImportPath)
		fmt.Fprintf(h, "pkg %x\ndriver %x\n", pkgID, plan.driverID)
		return h.Sum()
	}()
	typeErrsKey := func() cache.ActionID {
		h := cache.NewHash("analysis type errors " + a.Package.ImportPath)
		fmt.Fprintf(h, "types %x\n", typesKey)
		return h.Sum()
	}()
	passKey := func(index int) cache.ActionID {
		h := cache.NewHash("analysis pass " + a.Package.ImportPath)
		fmt.Fprintf(h, "pkg %x\nanalyzer %x\nreport %t\n", pkgID, plan.analyzers[index].id, reported(index))
		for _, h2 := range plan.analyzers[index].reqClosure {
			if plan.analyzers[h2].Facts {
				for _, a1 := range deps {
					fmt.Fprintf(h, "dep %q %x %x\n", a1.Package.ImportPath, plan.analyzers[h2].id, analysisResultOf(a1).passes[h2].factsID)
				}
			}
		}
		return h.Sum()
	}

	// Look up each entry; note the missing ones.
	c := cache.Default()
	get := func(key cache.ActionID) (analysisOutput, bool) {
		if !cfg.BuildA {
			if file, entry, err := cache.GetFile(c, key); err == nil {
				return analysisOutput{file: file, id: entry.OutputID}, true
			}
		}
		return analysisOutput{}, false
	}
	res := &analysisResult{passes: make(map[int]analysisOutput)}
	// The types and type errors are produced together.
	if out, ok := get(typesKey); ok {
		if data, _, err := cache.GetBytes(c, typeErrsKey); err == nil && json.Unmarshal(data, &res.typeErrs) == nil {
			res.types = out
		}
	}
	needTypes := res.types.file == ""
	var missing []int
	for _, index := range plan.need[a] {
		if out, ok := get(passKey(index)); ok && out.readPassInfo() == nil {
			res.passes[index] = out
		} else {
			missing = append(missing, index)
		}
	}

	// Run the tool, if necessary.
	if len(missing) > 0 || needTypes {
		areq := &protocol.AnalyzeRequest{
			ID:           vcfg0.ID,
			ImportPath:   vcfg0.ImportPath,
			Dir:          vcfg0.Dir,
			GoFiles:      goFiles,
			NonGoFiles:   nonGoFiles,
			IgnoredFiles: ignoredFiles,
			ImportMap:    vcfg0.ImportMap,
			GoVersion:    vcfg0.GoVersion,
			GOARCH:       cfg.Goarch,
			Module:       protocolModule(vcfg0.Module),
			TypeImports:  make(map[string]string),
			Facts:        make(map[int]map[string]string),
			Outputs:      make(map[int]string),
			Report:       []int{},
		}
		for _, a1 := range deps {
			areq.TypeImports[a1.Package.ImportPath] = analysisResultOf(a1).types.file
		}
		if needTypes {
			areq.TypeExport = a.Objdir + "types"
			areq.TypeErrors = a.Objdir + "typeerrors"
		}
		for i, index := range missing {
			areq.Outputs[index] = fmt.Sprintf("%spass%d", a.Objdir, i)
			if reported(index) {
				areq.Report = append(areq.Report, index)
			}
			for _, h2 := range plan.analyzers[index].reqClosure {
				if plan.analyzers[h2].Facts && areq.Facts[h2] == nil {
					files := make(map[string]string)
					for _, a1 := range deps {
						// A trivial pass file need not be read:
						// a missing entry denotes no facts.
						if out, ok := analysisResultOf(a1).passes[h2]; ok && !out.trivial {
							files[a1.Package.ImportPath] = out.file
						}
					}
					areq.Facts[h2] = files
				}
			}
		}

		sh := b.Shell(a)
		js, err := json.Marshal(areq)
		if err != nil {
			return fmt.Errorf("internal error marshaling analysis config: %v", err)
		}
		if err := sh.Mkdir(a.Objdir); err != nil {
			return err
		}
		if err := sh.writeFile(a.Objdir+"analysis.cfg", js); err != nil {
			return err
		}
		p := a.Package
		if err := sh.run(p.Dir, p.ImportPath, b.cCompilerEnv(), cfg.BuildToolexec, st.toolPath, a.Objdir+"analysis.cfg"); err != nil {
			return err
		}

		// Save the outputs in the cache.
		put := func(key cache.ActionID, file string) (analysisOutput, error) {
			f, err := os.Open(file)
			if err != nil {
				return analysisOutput{}, err
			}
			defer f.Close()
			id, _, err := c.Put(key, f)
			if err != nil {
				return analysisOutput{}, err
			}
			return analysisOutput{file: c.OutputFile(id), id: id}, nil
		}
		if needTypes {
			data, err := os.ReadFile(areq.TypeErrors)
			if err != nil {
				return err
			}
			if err := json.Unmarshal(data, &res.typeErrs); err != nil {
				return fmt.Errorf("decoding type errors of %s: %v", vcfg0.ID, err)
			}
			if err := cache.PutBytes(c, typeErrsKey, data); err != nil {
				return err
			}
			res.types, err = put(typesKey, areq.TypeExport)
			if err != nil {
				return err
			}
		}
		for _, index := range missing {
			out, err := put(passKey(index), areq.Outputs[index])
			if err != nil {
				return err
			}
			if err := out.readPassInfo(); err != nil {
				return err
			}
			res.passes[index] = out
		}
	}
	a.Provider = res

	// Report the type errors of this package and the failures of
	// analyzers on it, whether computed or cached. Each is
	// reported once, here; the passes skipped because of it (here
	// and in importers) are not.
	var buf bytes.Buffer
	for _, msg := range res.typeErrs {
		fmt.Fprintf(&buf, "%s\n", msg)
	}
	for _, index := range plan.need[a] {
		if f := res.passes[index].failure; f != nil {
			fmt.Fprintf(&buf, "%s\n", f)
		}
	}
	if buf.Len() > 0 {
		return b.Shell(a).reportCmd("", "", buf.Bytes(), errors.New("analysis failed"))
	}
	return nil
}

// analysisReport is the actor for an analysisreport action.
func (b *Builder) analysisReport(ctx context.Context, r *Action) error {
	a := r.Deps[0]
	if a.Actor == nil || a.Deps[0].Failed != nil {
		return nil // built-in package, or build failed (reported elsewhere)
	}
	plan := b.analysis.plan
	if plan == nil {
		return nil // tool or metadata query failed (reported elsewhere)
	}
	reported := plan.reported[a]

	var buf bytes.Buffer
	incomplete := false // some reported analyzer failed or was skipped

	// Report errors in the references.
	for _, msg := range plan.refErrs[a] {
		fmt.Fprintf(&buf, "%s\n", msg)
	}

	// Report the diagnostics of the reported analyzers,
	// ordered by name. (Type errors and failures are
	// reported by the analysisvet action of the
	// package in which they occur.)
	indexes := slices.SortedFunc(maps.Keys(reported), func(x, y int) int {
		return strings.Compare(reported[x].name, reported[y].name)
	})
	res := analysisResultOf(a)
	if res == nil {
		// Analysis of the package failed (reported elsewhere).
		incomplete = len(indexes) > 0
	} else {
		for _, index := range indexes {
			en := reported[index]
			out, ok := res.passes[index]
			if !ok {
				// Every reported analyzer is in plan.need[a].
				return fmt.Errorf("internal error: no pass file for analyzer %s", en.name)
			}
			pf, err := protocol.ReadPass(out.file)
			if err != nil {
				return err
			}
			if pf.Failure != nil || pf.Skipped {
				incomplete = true
			}
			for _, d := range pf.Diagnostics {
				fmt.Fprintf(&buf, "%s: %s\n", d.Posn, d.Message)
			}
		}
	}
	if buf.Len() > 0 {
		return b.Shell(r).reportCmd("", "", buf.Bytes(), errors.New("analysis reported problems"))
	}

	if incomplete {
		// Some analyzer failed or was skipped, because of
		// an error reported elsewhere (see analysisVet).
		// The test must not run, but there is nothing more
		// to say: this action fails as a consequence of the
		// analysis action, without an error of its own.
		r.Failed = a
	}

	return nil
}

// readPassInfo reads the pass file out.file, written by the analysis
// tool, and sets the other fields of out from it.
//
// factsID is the hash of the parts of the pass file other than its
// diagnostics: its facts, its failure, and whether it was skipped. It
// is the part of the pass file on which dependent passes depend.
//
// trivial reports whether the pass file has no facts, no failure, and
// was not skipped. The analysis of an importer need not read a
// trivial pass file.
func (out *analysisOutput) readPassInfo() error {
	pf, err := protocol.ReadPass(out.file)
	if err != nil {
		return err
	}
	h := sha256.New()
	fmt.Fprintf(h, "facts %d\n", len(pf.Facts))
	h.Write(pf.Facts)
	if f := pf.Failure; f != nil {
		fmt.Fprintf(h, "failure %q %q %q\n", f.Package, f.Analyzer, f.Message)
	}
	fmt.Fprintf(h, "skipped %t\n", pf.Skipped)
	h.Sum(out.factsID[:0])
	out.trivial = len(pf.Facts) == 0 && pf.Failure == nil && !pf.Skipped
	out.failure = pf.Failure
	return nil
}

// protocolModule converts the module of a vet configuration to its
// form in the analysis protocol.
func protocolModule(m *vetModule) *protocol.Module {
	if m == nil {
		return nil
	}
	pm := &protocol.Module{
		Path:      m.Path,
		Version:   m.Version,
		Replace:   protocolModule(m.Replace),
		Time:      m.Time,
		Main:      m.Main,
		Indirect:  m.Indirect,
		Dir:       m.Dir,
		GoMod:     m.GoMod,
		GoVersion: m.GoVersion,
	}
	if m.Error != nil {
		pm.Error = &protocol.ModuleError{Err: m.Error.Err}
	}
	return pm
}
