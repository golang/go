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
// facts) it expands suites, applies selectors, and computes, for
// each root package, the set of analyzers whose diagnostics are
// reported, and for each analysisvet action, the set of analyzers
// whose results it must produce. An analysisvet action whose set is
// empty merely produces types.
//
// Within the tool, analyzers are identified by pointer; in the
// protocol, they are identified by their index in the tool's
// metadata, a topological order that is deterministic for a given
// tool.
//
// Each analysisvet action produces, for each needed analyzer, one
// pass file containing that analyzer's facts (if it uses facts) and,
// for the analyzers reported for a root package, its diagnostics.
// So the facts of an analyzer required by several roots are computed
// and stored once per package. A pass file may instead record a
// failure, which prevents dependent passes and is reported at each
// affected root package.
//
// For now, analysis results are not cached: the tool runs on every
// package in the closure of the root packages, writing its outputs
// to the action's object directory.

import (
	"bytes"
	"context"
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

	"cmd/go/internal/cfg"
	"cmd/go/internal/load"
	"cmd/go/internal/modload"
)

// analysisResult records the outputs of an analysisvet action.
// It is the action's Provider.
type analysisResult struct {
	types    analysisOutput         // the package's types
	typeErrs []string               // type errors of the package (reported once)
	passes   map[int]analysisOutput // analyzer index -> pass file
}

// An analysisOutput is an output file of the analysis tool plus, for
// a pass file, information read from it (see readPassInfo).
type analysisOutput struct {
	file    string
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

	toolPath string // the executable of the tool

	// plan is set by the analysismeta action, before any
	// analysisvet action runs, if the metadata is available.
	plan *analysisPlan
}

// An analysisPlan holds the metadata of the analysis tool's analyzers
// (by index) and the plan derived from it.
type analysisPlan struct {
	analyzers []analyzerInfo                        // index -> analyzer
	need      map[*Action][]int                     // analysisvet action -> sorted indexes of needed pass files
	reported  map[*Action]map[int]*reportedAnalyzer // root analysisvet action -> index -> analyzer whose diagnostics are reported
	refErrs   map[*Action][]string                  // root analysisvet action -> errors in its references
}

// An analyzerInfo holds the metadata of one of the analysis tool's
// analyzers and information derived from it.
type analyzerInfo struct {
	*protocol.Analyzer       // metadata (Name, Requires, Facts, etc.)
	reqClosure         []int // sorted indexes of Requires closure (inclusive)
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
	data, err := b.analysisMetadata(a)
	if err != nil || data == nil {
		return err // (data is nil with -n)
	}
	md := new(protocol.Metadata)
	if err := json.Unmarshal(data, md); err != nil {
		return fmt.Errorf("decoding analysis tool metadata: %v", err)
	}
	b.planAnalysis(md)
	return nil
}

// analysisMetadata runs the tool's metadata query and returns its
// result. It returns nil with -n.
func (b *Builder) analysisMetadata(a *Action) ([]byte, error) {
	st := &b.analysis
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
	return os.ReadFile(file)
}

// planAnalysis performs the dynamic stage of resolution, computing
// the analyzers reported for each root package and the set of pass
// files needed from each analysisvet action. It sets st.plan.
func (b *Builder) planAnalysis(md *protocol.Metadata) {
	st := &b.analysis

	// Index the metadata. Analyzers are identified by their index
	// in md.Analyzers, which is in topological order (prerequisites
	// first).
	plan := &analysisPlan{
		analyzers: make([]analyzerInfo, len(md.Analyzers)),
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

	// Expand suites, apply selectors, and compute the analyzers
	// reported for each root package.
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
// the tool's metadata, it expands suites and applies selectors. It
// returns the analyzers whose diagnostics are reported, by index, and
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

	// Run the tool.
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
		TypeExport:   a.Objdir + "types",
		TypeErrors:   a.Objdir + "typeerrors",
		Facts:        make(map[int]map[string]string),
		Outputs:      make(map[int]string),
		Report:       []int{},
	}
	for _, a1 := range deps {
		areq.TypeImports[a1.Package.ImportPath] = analysisResultOf(a1).types.file
	}
	for i, index := range plan.need[a] {
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

	// Read the outputs.
	res := &analysisResult{passes: make(map[int]analysisOutput)}
	data, err := os.ReadFile(areq.TypeErrors)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(data, &res.typeErrs); err != nil {
		return fmt.Errorf("decoding type errors of %s: %v", vcfg0.ID, err)
	}
	res.types = analysisOutput{file: areq.TypeExport}
	for _, index := range plan.need[a] {
		out := analysisOutput{file: areq.Outputs[index]}
		if err := out.readPassInfo(); err != nil {
			return err
		}
		res.passes[index] = out
	}
	a.Provider = res

	// Report the type errors of this package and
	// the failures of analyzers on it. Each is
	// reported once, here; the passes skipped because
	// of it (here and in importers) are not.
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
// trivial reports whether the pass file has no facts, no failure, and
// was not skipped. The analysis of an importer need not read a
// trivial pass file.
func (out *analysisOutput) readPassInfo() error {
	pf, err := protocol.ReadPass(out.file)
	if err != nil {
		return err
	}
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
