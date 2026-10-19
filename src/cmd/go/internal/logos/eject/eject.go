// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package eject implements logos eject: it writes a complete pure-Go
// copy of the main module for upstream Go toolchains. Every .lgo file
// becomes a .go file compiled from the same lowered nodes the native
// build uses, every ejected loGo/s dependency module lands under
// DIR/_logos_deps, and the support sources become the shared module
// github.com/gosuda/loGos/rt.
package eject

import (
	"bytes"
	"context"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"

	"golang.org/x/mod/modfile"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/load"
	"cmd/go/internal/modload"
	"cmd/go/internal/work"
)

// rtModulePath is the shared support module for ejected trees. Every
// ejected loGo/s dependency keeps one type identity for Option and
// friends through it.
const rtModulePath = "github.com/gosuda/loGos/rt"

// rtRequireVersion pins the support module in ejected go.mod files.
// The replace line always points at the ejected copy, so the version
// never resolves over the network.
const rtRequireVersion = "v0.0.0"

// CmdEject implements the logos eject command.
var CmdEject = &base.Command{
	Run:       runEject,
	UsageLine: "logos eject [-o DIR]",
	Short:     "write a pure-Go copy of the main module",
	Long: `
Eject writes a complete pure-Go copy of the main module to DIR:
every .lgo file becomes a .go file with comments kept, every
ejected loGo/s dependency module lands under DIR/_logos_deps, and
the support sources become the shared module
github.com/gosuda/loGos/rt under DIR/_logos_deps/rt.

DIR must not exist or must be empty. The ejected tree builds with an
upstream Go 1.27 toolchain. Eject never writes into the source tree.
	`,
}

// ejectOut holds the required -o DIR output directory.
var ejectOut string

// EmitGo holds the logos build --emit-go directory. When set, logos
// build is an alias of logos eject into that directory.
var EmitGo string

func init() {
	CmdEject.Flag.StringVar(&ejectOut, "o", "", "")
	base.AddChdirFlag(&CmdEject.Flag)
}

// Register adds the eject subcommand and the logos-only --emit-go build
// flag to the logos root command.
func Register(root *base.Command) {
	root.Commands = append(root.Commands, CmdEject)
	build := root.Lookup("build")
	if build == nil {
		return
	}
	if build.Flag.Lookup("emit-go") == nil {
		build.Flag.StringVar(&EmitGo, "emit-go", "", "")
	}
	orig := build.Run
	build.Run = func(ctx context.Context, cmd *base.Command, args []string) {
		if EmitGo == "" {
			orig(ctx, cmd, args)
			return
		}
		ejectToDir(ctx, EmitGo)
	}
	const help = "\n\t-emit-go=DIR\n\t\twrite a pure-Go copy of the main module to DIR instead of building (logos builds only).\n"
	if !strings.Contains(build.Long, "-emit-go") {
		before, after, found := strings.Cut(build.Long, "\nSee also:")
		if found {
			build.Long = before + help + "\nSee also:" + after
			return
		}
		build.Long += help
	}
}

// runEject implements the logos eject command. Eject covers the whole
// main module, so package arguments are rejected.
func runEject(ctx context.Context, cmd *base.Command, args []string) {
	if len(args) > 0 {
		base.Fatalf("logos eject: unexpected package arguments")
	}
	ejectToDir(ctx, ejectOut)
}

// ejectModule groups the ejected packages of one module.
type ejectModule struct {
	path    string
	version string
	dir     string
	pkgs    []*load.Package
}

// ejectPlan splits the reachable packages into the main module, the
// ejected loGo/s dependency modules, and the emission targets.
type ejectPlan struct {
	main      *ejectModule
	deps      []*ejectModule
	depByPath map[string]*ejectModule
	logos     []*load.Package
	known     map[*load.Package]bool
	testBuilt map[*load.Package]bool
}

// testBuild holds the test variant of one package for emission of its
// test files.
type testBuild struct {
	main   *load.Package
	ptest  *load.Package
	pxtest *load.Package
}

// ejectToDir writes the ejected tree to out. It loads the main module,
// compiles every loGo/s package with -logos.eject into staging, then
// assembles out from the staging output plus copied sources. The two
// builds run with ModeBuild only, so link outputs land under the build
// work directory and nothing is written into the source tree.
func ejectToDir(ctx context.Context, out string) {
	checkEjectOut(out)
	moduleLoader := modload.NewLoader()
	moduleLoader.InitWorkfile()
	work.BuildInit(moduleLoader)
	staging, err := os.MkdirTemp("", "logoseject")
	if err != nil {
		base.Fatalf("logos eject: %v", err)
	}
	defer os.RemoveAll(staging)
	load.ArmLogosEject(staging)
	defer load.DisarmLogosEject()
	opts := load.PackageOpts{AutoVCS: true}
	pkgs := load.PackagesAndErrors(moduleLoader, ctx, opts, []string{"all"})
	load.CheckPackageErrors(pkgs)
	plan := arrangeEjectPlan(pkgs)
	builds := collectTestBuilds(moduleLoader, ctx, opts, plan)
	makeEjectStagingDirs(plan, staging, builds)
	buildEjectRoots(moduleLoader, ctx, phaseOneRoots(plan))
	if len(builds) > 0 {
		stripEjectCopies(builds)
		buildEjectRoots(moduleLoader, ctx, testBuildMains(builds))
	}
	writeEjectedTree(plan, staging, out)
}

// checkEjectOut rejects an output directory that exists and is not
// empty. It runs before any loading or building, so a rejected eject
// costs nothing and changes nothing.
func checkEjectOut(out string) {
	if out == "" {
		base.Fatalf("logos eject: -o DIR is required")
	}
	st, err := os.Stat(out)
	if err != nil {
		if os.IsNotExist(err) {
			return
		}
		base.Fatalf("logos eject: %s: %v", out, err)
	}
	if !st.IsDir() {
		base.Fatalf("logos eject: %s: not a directory", out)
	}
	entries, err := os.ReadDir(out)
	if err != nil {
		base.Fatalf("logos eject: %s: %v", out, err)
	}
	if len(entries) > 0 {
		base.Fatalf("logos eject: %s: directory is not empty", out)
	}
}

// arrangeEjectPlan groups the packages reachable from the main module
// into the main module entry, the loGo/s dependency modules, and the
// emission targets.
func arrangeEjectPlan(pkgs []*load.Package) *ejectPlan {
	gomod := modload.FindGoMod(base.Cwd())
	if gomod == "" {
		base.Fatalf("logos eject: no main module (run inside a module)")
	}
	plan := &ejectPlan{
		main:      &ejectModule{dir: filepath.Dir(gomod)},
		depByPath: map[string]*ejectModule{},
		known:     map[*load.Package]bool{},
		testBuilt: map[*load.Package]bool{},
	}
	for _, p := range load.PackageList(pkgs) {
		plan.addPackage(p)
	}
	resortEjectPlan(plan)
	return plan
}

// addPackage files one reachable package into the plan. Test artifacts
// stay out: test mains and test variants build in the second phase,
// never as plan roots.
func (plan *ejectPlan) addPackage(p *load.Package) {
	if plan.known[p] || p.Standard || p.Module == nil || isTestArtifact(p) {
		return
	}
	plan.known[p] = true
	if p.Module.Main {
		if p.Module.Dir != "" && p.Module.Dir != plan.main.dir {
			base.Fatalf("logos eject: workspaces with several main modules are not supported")
		}
		plan.main.pkgs = append(plan.main.pkgs, p)
	} else {
		m := plan.depByPath[p.Module.Path]
		if m == nil {
			if p.Module.Path == "" || p.Module.Version == "" {
				base.Fatalf("logos eject: dependency module has no path or version")
			}
			if p.Module.Dir == "" {
				base.Fatalf("logos eject: module %s has no directory", p.Module.Path)
			}
			m = &ejectModule{path: p.Module.Path, version: p.Module.Version, dir: p.Module.Dir}
			plan.depByPath[p.Module.Path] = m
		}
		if m.version != p.Module.Version || m.dir != p.Module.Dir {
			base.Fatalf("logos eject: module %s has several versions", p.Module.Path)
		}
		m.pkgs = append(m.pkgs, p)
	}
	if p.IsLogos() {
		plan.logos = append(plan.logos, p)
	}
}

// resortEjectPlan rebuilds the sorted dependency and target lists after
// test-only dependencies joined the plan.
func resortEjectPlan(plan *ejectPlan) {
	plan.deps = nil
	for _, m := range plan.depByPath {
		if moduleHasLogos(m) {
			plan.deps = append(plan.deps, m)
		}
	}
	slices.SortFunc(plan.deps, func(a, b *ejectModule) int { return strings.Compare(a.path, b.path) })
	slices.SortFunc(plan.logos, func(a, b *load.Package) int { return strings.Compare(a.ImportPath, b.ImportPath) })
	slices.SortFunc(plan.main.pkgs, func(a, b *load.Package) int { return strings.Compare(a.ImportPath, b.ImportPath) })
}

// moduleHasLogos reports whether any package of m is a loGo/s package.
func moduleHasLogos(m *ejectModule) bool {
	for _, p := range m.pkgs {
		if p.IsLogos() {
			return true
		}
	}
	return false
}

// isTestArtifact reports whether p is a synthesized test main or a test
// variant produced for one. Such packages build in the second phase.
func isTestArtifact(p *load.Package) bool {
	return p.ForTest != "" || p.Internal.TestmainGo != nil
}

// hasLogosTestFiles reports whether p holds loGo/s test files needing
// test-variant compilation for emission.
func hasLogosTestFiles(p *load.Package) bool {
	for _, name := range p.TestGoFiles {
		if cfg.IsLogosFile(name) {
			return true
		}
	}
	for _, name := range p.XTestGoFiles {
		if cfg.IsLogosFile(name) {
			return true
		}
	}
	return false
}

// collectTestBuilds prepares test-variant builds for every loGo/s
// package holding loGo/s test files, and absorbs the test-only
// dependencies their loads discovered into the plan.
func collectTestBuilds(moduleLoader *modload.Loader, ctx context.Context, opts load.PackageOpts, plan *ejectPlan) []testBuild {
	var builds []testBuild
	var bad []*load.Package
	for _, p := range plan.logos {
		if !hasLogosTestFiles(p) {
			continue
		}
		pmain, ptest, pxtest, perr := load.TestPackagesFor(moduleLoader, ctx, opts, p, nil)
		if perr != nil {
			bad = append(bad, perr)
			continue
		}
		if pmain == nil {
			base.Fatalf("logos eject: no test main for %s", p.ImportPath)
		}
		builds = append(builds, testBuild{main: pmain, ptest: ptest, pxtest: pxtest})
		plan.testBuilt[p] = true
	}
	load.CheckPackageErrors(bad)
	var mains []*load.Package
	for _, t := range builds {
		mains = append(mains, t.main)
	}
	for _, p := range load.PackageList(mains) {
		plan.addPackage(p)
	}
	resortEjectPlan(plan)
	return builds
}

// makeEjectStagingDirs creates one staging directory per emission
// target, including test variants, so the compiler always finds its
// -logos.eject directory.
func makeEjectStagingDirs(plan *ejectPlan, staging string, builds []testBuild) {
	mkdir := func(importPath string) {
		if err := os.MkdirAll(filepath.Join(staging, filepath.FromSlash(importPath)), 0777); err != nil {
			base.Fatalf("logos eject: %v", err)
		}
	}
	for _, p := range plan.logos {
		mkdir(p.ImportPath)
	}
	for _, t := range builds {
		for _, p := range []*load.Package{t.ptest, t.pxtest} {
			if p != nil {
				mkdir(p.ImportPath)
			}
		}
	}
}

// testBuildMains returns the test main roots of the second phase.
func testBuildMains(builds []testBuild) []*load.Package {
	mains := make([]*load.Package, 0, len(builds))
	for _, t := range builds {
		mains = append(mains, t.main)
	}
	return mains
}

// phaseOneRoots returns the first-phase build roots: every main-module
// package plus every marked loGo/s package, sorted for determinism.
func phaseOneRoots(plan *ejectPlan) []*load.Package {
	set := map[*load.Package]bool{}
	var roots []*load.Package
	for _, p := range plan.main.pkgs {
		if p.Standard || set[p] {
			continue
		}
		set[p] = true
		roots = append(roots, p)
	}
	for _, p := range plan.logos {
		if set[p] {
			continue
		}
		set[p] = true
		roots = append(roots, p)
	}
	return roots
}

// buildEjectRoots compiles roots and their dependencies with ModeBuild.
// Link outputs land under the build work directory, never the source
// tree. A failed build exits before any output is written.
func buildEjectRoots(moduleLoader *modload.Loader, ctx context.Context, roots []*load.Package) {
	b := work.NewBuilder("", moduleLoader.VendorDirOrEmpty)
	defer func() {
		if err := b.Close(); err != nil {
			base.Fatalf("logos eject: %v", err)
		}
	}()
	root := &work.Action{Mode: "go build"}
	for _, p := range roots {
		root.Deps = append(root.Deps, b.AutoAction(moduleLoader, work.ModeBuild, work.ModeBuild, p))
	}
	b.Do(ctx, root)
	if root.Failed != nil {
		base.SetExitStatus(1)
	}
	if base.GetExitStatus() != 0 {
		base.Exit()
	}
}

// stripEjectCopies keeps emission on exactly one compile per package in
// the second phase: the test variants. Test copies of dependencies lose
// the flag, so no staging file gets two concurrent writers.
func stripEjectCopies(builds []testBuild) {
	keep := map[*load.Package]bool{}
	var mains []*load.Package
	for _, t := range builds {
		mains = append(mains, t.main)
		if t.ptest != nil {
			keep[t.ptest] = true
			t.ptest.Internal.LogosEject = true
		}
		if t.pxtest != nil {
			keep[t.pxtest] = true
			t.pxtest.Internal.LogosEject = true
		}
	}
	for _, p := range load.PackageList(mains) {
		if p.Internal.LogosEject && !keep[p] {
			p.Internal.LogosEject = false
			p.Internal.Gcflags = withoutEjectFlags(p.Internal.Gcflags)
		}
	}
}

// withoutEjectFlags returns flags without the -logos.eject entries in a
// fresh slice, leaving shared backing arrays untouched.
func withoutEjectFlags(flags []string) []string {
	out := make([]string, 0, len(flags))
	for _, f := range flags {
		if !strings.HasPrefix(f, "-logos.eject=") {
			out = append(out, f)
		}
	}
	return out
}

// writeEjectedTree assembles the output directory from copied sources,
// compiler-emitted files, the support module, and rewritten go.mod files.
func writeEjectedTree(plan *ejectPlan, staging, out string) {
	abs, err := filepath.Abs(out)
	if err != nil {
		base.Fatalf("logos eject: %s: %v", out, err)
	}
	copyModuleTree(plan.main.dir, out, ejectSkipDir(plan.main.dir, abs, out))
	usedRT := false
	for _, p := range plan.main.pkgs {
		if emitPackageGoFiles(p, plan, staging, out, plan.main.dir) {
			usedRT = true
		}
	}
	ejected := map[string]string{}
	for _, m := range plan.deps {
		dst := filepath.Join(out, "_logos_deps", m.path+"@"+m.version)
		ejected[m.path] = dst
		copyModuleTree(m.dir, dst, ejectSkipDir(m.dir, abs, out))
		for _, p := range m.pkgs {
			if emitPackageGoFiles(p, plan, staging, dst, m.dir) {
				usedRT = true
			}
		}
	}
	var rtDir string
	if usedRT {
		rtDir = filepath.Join(out, "_logos_deps", "rt")
		writeSupportModule(out, goVersionOf(parseGoModFile(filepath.Join(out, "go.mod"))))
		addSupportRequire(filepath.Join(out, "go.mod"), "./_logos_deps/rt")
	}
	rewriteMainGoMod(out, plan)
	for _, m := range plan.deps {
		rewriteDepGoMod(plan, m, ejected)
	}
	if usedRT {
		for _, m := range plan.deps {
			addSupportRequire(filepath.Join(ejected[m.path], "go.mod"), relEjectTarget(ejected[m.path], rtDir))
		}
	}
}

// emitPackageGoFiles writes the compiler-emitted Go files of one loGo/s
// package under dstRoot, rewritten to the ejected import paths and
// formatted. It reports whether any file uses the support module.
func emitPackageGoFiles(p *load.Package, plan *ejectPlan, staging, dstRoot, modRoot string) bool {
	if !p.IsLogos() {
		return false
	}
	rel, err := filepath.Rel(modRoot, p.Dir)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		base.Fatalf("logos eject: package %s outside module", p.ImportPath)
	}
	names := slices.Clone(p.GoFiles)
	if plan.testBuilt[p] {
		names = append(names, p.TestGoFiles...)
		names = append(names, p.XTestGoFiles...)
	}
	used := false
	for _, name := range names {
		if !cfg.IsLogosFile(name) {
			continue
		}
		stem := logosStem(name)
		raw, err := os.ReadFile(filepath.Join(staging, filepath.FromSlash(p.ImportPath), stem+".go"))
		if err != nil {
			base.Fatalf("logos eject: missing ejected output for %s", filepath.Join(p.Dir, name))
		}
		fixed, uses := rewriteEjectedGo(raw, name)
		if uses {
			used = true
		}
		dst := filepath.Join(dstRoot, rel, stem+".go")
		if err := os.MkdirAll(filepath.Dir(dst), 0777); err != nil {
			base.Fatalf("logos eject: %v", err)
		}
		if err := os.WriteFile(dst, fixed, 0666); err != nil {
			base.Fatalf("logos eject: %v", err)
		}
	}
	return used
}

// logosStem returns name without its loGo/s extension, keeping a _test
// suffix when present.
func logosStem(name string) string {
	if i := strings.LastIndex(name, "."); i >= 0 {
		return name[:i]
	}
	return name
}

// ejectSkipDir returns the path of out relative to one source module
// when out sits inside it, so the module copy skips the destination
// instead of feeding ejected files back into the tree.
func ejectSkipDir(modDir, outAbs, out string) string {
	rel, err := filepath.Rel(modDir, outAbs)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return ""
	}
	if rel == "." {
		base.Fatalf("logos eject: %s: output is the module directory", out)
	}
	return rel
}

// copyModuleTree copies the non-loGo/s files of one module: every
// regular file except .git contents, logos.toml files, loGo/s sources
// (emitted separately), and the skip subtree (the destination when it
// sits inside the module). Symlinks and special files stay out, so a
// link can never pull outside bytes into the tree.
func copyModuleTree(srcRoot, dstRoot, skip string) {
	err := filepath.WalkDir(srcRoot, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(srcRoot, path)
		if err != nil {
			return err
		}
		if rel == "." {
			return os.MkdirAll(dstRoot, 0777)
		}
		if skip != "" && (rel == skip || strings.HasPrefix(rel, skip+string(filepath.Separator))) {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		for seg := range strings.SplitSeq(rel, string(filepath.Separator)) {
			if seg == ".git" {
				if d.IsDir() {
					return filepath.SkipDir
				}
				return nil
			}
		}
		if d.IsDir() {
			return os.MkdirAll(filepath.Join(dstRoot, rel), 0777)
		}
		if !d.Type().IsRegular() {
			return nil
		}
		name := filepath.Base(rel)
		if name == "logos.toml" || cfg.IsLogosFile(name) {
			return nil
		}
		return copyDataFile(path, filepath.Join(dstRoot, rel))
	})
	if err != nil {
		base.Fatalf("logos eject: %v", err)
	}
}

// copyDataFile copies one non-loGo/s file. Go files that still parse
// are rewritten to the ejected import paths and formatted, so the tree
// stays gofmt-clean; anything else rides along byte for byte.
func copyDataFile(src, dst string) error {
	data, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	if strings.HasSuffix(src, ".go") {
		if fixed, _, ok := convertGoFile(data); ok {
			data = fixed
		}
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0777); err != nil {
		return err
	}
	return os.WriteFile(dst, data, 0666)
}

// rewriteEjectedGo rewrites one compiler-emitted file to the ejected
// import paths and formats it. Compiler output must parse; anything
// else is a loud failure, never a silent copy.
func rewriteEjectedGo(data []byte, name string) ([]byte, bool) {
	fixed, uses, ok := convertGoFile(data)
	if !ok {
		base.Fatalf("logos eject: cannot parse ejected file for %s", name)
	}
	return fixed, uses
}

// convertGoFile parses one Go file, remaps loGo/s support imports,
// drops comptime imports, sorts the import blocks, and formats the
// result. It reports whether any support import was used.
func convertGoFile(data []byte) ([]byte, bool, bool) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", data, parser.ParseComments)
	if err != nil {
		return nil, false, false
	}
	uses := remapLogosImports(f)
	ast.SortImports(fset, f)
	var buf bytes.Buffer
	if err := format.Node(&buf, fset, f); err != nil {
		return nil, false, false
	}
	return buf.Bytes(), uses, true
}

// remapLogosImports rewrites logos/rt and logos/sync imports to the
// shared ejected module and drops comptime imports. Empty import
// declarations fall away.
func remapLogosImports(f *ast.File) bool {
	uses := false
	var decls []ast.Decl
	for _, decl := range f.Decls {
		gen, ok := decl.(*ast.GenDecl)
		if !ok || gen.Tok != token.IMPORT {
			decls = append(decls, decl)
			continue
		}
		var specs []ast.Spec
		for _, spec := range gen.Specs {
			imp, ok := spec.(*ast.ImportSpec)
			if !ok {
				specs = append(specs, spec)
				continue
			}
			path, err := strconv.Unquote(imp.Path.Value)
			if err != nil {
				specs = append(specs, spec)
				continue
			}
			if path == "comptime" {
				continue
			}
			if rest, ok := strings.CutPrefix(path, "logos/"); ok {
				rest, _ = strings.CutPrefix(rest, "rt/")
				if rest == "rt" {
					imp.Path.Value = strconv.Quote(rtModulePath)
				} else {
					imp.Path.Value = strconv.Quote(rtModulePath + "/" + rest)
				}
				uses = true
			}
			specs = append(specs, spec)
		}
		if len(specs) == 0 {
			continue
		}
		gen.Specs = specs
		decls = append(decls, decl)
	}
	f.Decls = decls
	return uses
}

// writeSupportModule writes the shared support module: src/logos/rt at
// the root, src/logos/sync as sync/, without runtime and test files
// and without ejected build lines.
func writeSupportModule(out, goVer string) {
	goroot := runtime.GOROOT()
	rtDst := filepath.Join(out, "_logos_deps", "rt")
	copySupportDir(filepath.Join(goroot, "src", "logos", "rt"), rtDst)
	if st, err := os.Stat(filepath.Join(goroot, "src", "logos", "sync")); err == nil && st.IsDir() {
		copySupportDir(filepath.Join(goroot, "src", "logos", "sync"), filepath.Join(rtDst, "sync"))
	}
	if err := os.WriteFile(filepath.Join(rtDst, "go.mod"), []byte("module "+rtModulePath+"\n\ngo "+goVer+"\n"), 0666); err != nil {
		base.Fatalf("logos eject: %v", err)
	}
}

// copySupportDir copies the Go sources of one support directory,
// skipping runtime variants, test files, and non-Go files.
func copySupportDir(src, dst string) {
	entries, err := os.ReadDir(src)
	if err != nil {
		base.Fatalf("logos eject: %v", err)
	}
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() {
			if strings.HasPrefix(name, ".") || name == "testdata" {
				continue
			}
			copySupportDir(filepath.Join(src, name), filepath.Join(dst, name))
			continue
		}
		if !strings.HasSuffix(name, ".go") {
			continue
		}
		if strings.HasSuffix(name, "_runtime.go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(src, name))
		if err != nil {
			base.Fatalf("logos eject: %v", err)
		}
		fixed, _, ok := convertGoFile(stripEjectedBuildTag(data))
		if !ok {
			base.Fatalf("logos eject: cannot parse support file %s", name)
		}
		if err := os.MkdirAll(dst, 0777); err != nil {
			base.Fatalf("logos eject: %v", err)
		}
		if err := os.WriteFile(filepath.Join(dst, name), fixed, 0666); err != nil {
			base.Fatalf("logos eject: %v", err)
		}
	}
}

// stripEjectedBuildTag removes ejected build lines, so the support copy
// compiles under plain go with no tags set.
func stripEjectedBuildTag(data []byte) []byte {
	lines := bytes.Split(data, []byte("\n"))
	kept := lines[:0]
	for _, l := range lines {
		trimmed := bytes.TrimSpace(l)
		if bytes.Equal(trimmed, []byte("//go:build logosejected")) || bytes.Equal(trimmed, []byte("// +build logosejected")) {
			continue
		}
		kept = append(kept, l)
	}
	return bytes.Join(kept, []byte("\n"))
}

// parseGoModFile reads and parses one go.mod file.
func parseGoModFile(path string) *modfile.File {
	data, err := os.ReadFile(path)
	if err != nil {
		base.Fatalf("logos eject: %v", err)
	}
	f, err := modfile.Parse(path, data, nil)
	if err != nil {
		base.Fatalf("logos eject: %v", err)
	}
	return f
}

// writeGoModFile formats one go.mod file back to disk.
func writeGoModFile(path string, f *modfile.File) {
	if err := os.WriteFile(path, modfile.Format(f.Syntax), 0666); err != nil {
		base.Fatalf("logos eject: %v", err)
	}
}

// goVersionOf returns the go directive of f for the support go.mod.
func goVersionOf(f *modfile.File) string {
	if f.Go != nil && f.Go.Version != "" {
		return f.Go.Version
	}
	return "1.27"
}

// rewriteMainGoMod points the ejected main go.mod at every ejected
// dependency module with require plus replace lines.
func rewriteMainGoMod(out string, plan *ejectPlan) {
	path := filepath.Join(out, "go.mod")
	f := parseGoModFile(path)
	for _, m := range plan.deps {
		ensureGoRequire(f, m.path, m.version)
		setGoReplace(f, m.path, m.version, "./_logos_deps/"+m.path+"@"+m.version)
	}
	writeGoModFile(path, f)
}

// addSupportRequire points one ejected go.mod at the support module
// with require plus replace lines.
func addSupportRequire(gomodPath, target string) {
	f := parseGoModFile(gomodPath)
	ensureGoRequire(f, rtModulePath, rtRequireVersion)
	setGoReplace(f, rtModulePath, rtRequireVersion, target)
	writeGoModFile(gomodPath, f)
}

// relEjectTarget returns the go.mod replace target from one ejected
// directory to another.
func relEjectTarget(from, to string) string {
	rel, err := filepath.Rel(from, to)
	if err != nil {
		base.Fatalf("logos eject: %v", err)
	}
	return "./" + filepath.ToSlash(rel)
}

// rewriteDepGoMod points one ejected dependency go.mod at every other
// ejected dependency module, and repoints its own replaces of ejected
// modules at the ejected copies.
func rewriteDepGoMod(plan *ejectPlan, m *ejectModule, ejected map[string]string) {
	dst := ejected[m.path]
	f := parseGoModFile(filepath.Join(dst, "go.mod"))
	for _, other := range plan.deps {
		if other.path == m.path {
			continue
		}
		target := relEjectTarget(dst, ejected[other.path])
		ensureGoRequire(f, other.path, other.version)
		setGoReplace(f, other.path, other.version, target)
	}
	var olds [][2]string
	for _, r := range f.Replace {
		if _, ok := ejected[r.Old.Path]; ok {
			olds = append(olds, [2]string{r.Old.Path, r.Old.Version})
		}
	}
	for _, o := range olds {
		setGoReplace(f, o[0], o[1], relEjectTarget(dst, ejected[o[0]]))
	}
	writeGoModFile(filepath.Join(dst, "go.mod"), f)
}

// ensureGoRequire adds a require line unless one already names path.
func ensureGoRequire(f *modfile.File, path, vers string) {
	for _, r := range f.Require {
		if r.Mod.Path == path {
			return
		}
	}
	if err := f.AddRequire(path, vers); err != nil {
		base.Fatalf("logos eject: %v", err)
	}
}

// setGoReplace points every replace of oldPath at newPath, dropping
// stale targets first so no duplicate replaces remain.
func setGoReplace(f *modfile.File, oldPath, oldVers, newPath string) {
	for _, r := range slices.Clone(f.Replace) {
		if r.Old.Path == oldPath && (r.New.Path != newPath || r.New.Version != "") {
			if err := f.DropReplace(r.Old.Path, r.Old.Version); err != nil {
				base.Fatalf("logos eject: %v", err)
			}
		}
	}
	for _, r := range f.Replace {
		if r.Old.Path == oldPath && r.New.Path == newPath && r.New.Version == "" {
			return
		}
	}
	if err := f.AddReplace(oldPath, oldVers, newPath, ""); err != nil {
		base.Fatalf("logos eject: %v", err)
	}
}
