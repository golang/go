// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eject

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"go/ast"
	"go/build/constraint"
	"go/parser"
	"go/token"
	"internal/logosfile"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/load"
	"cmd/internal/logosowned"
)

func emitOwnedHelpers(ctx context.Context, plan *ejectPlan, staging, out string, ejected map[string]string) (bool, error) {
	reqs, err := gatherOwnedRequests(staging)
	if err != nil {
		return false, err
	}
	if len(reqs) == 0 {
		return false, nil
	}
	abs, err := filepath.Abs(out)
	if err != nil {
		return false, fmt.Errorf("resolve output: %v", err)
	}

	dests := make([]*ownedDest, 0, len(reqs))
	copiedRT := false
	for i := range reqs {
		req := &reqs[i]
		owner := findOwnedOwner(plan, req.Owner)
		if owner != nil && owner.Name != req.Package {
			// Platform-selected source files may use different package names.
			// A request must match an actual load, not just the host inventory.
			if variant := selectOwnedPackage(plan.provenance[req.Owner], req.Owner, req.Package); variant != nil {
				owner = variant
			}
		}
		if owner == nil {
			return false, fmt.Errorf("owned request for %q has no ejected destination (standard library packages are not writable)", req.Owner)
		}
		if owner.Standard || owner.Goroot || gorootContains(owner.Dir) {
			return false, fmt.Errorf("owned request for %q has no ejected destination (standard library packages are not writable)", req.Owner)
		}
		if owner.Module == nil {
			return false, fmt.Errorf("owned request for %q has no module", req.Owner)
		}
		if owner.Name == "" || req.Package != owner.Name {
			return false, fmt.Errorf("owned request package %q does not match owner package %q (%s)", req.Package, owner.Name, req.Owner)
		}
		dstRoot, modRoot, copyUsesRT := ownedOutputRoots(plan, owner, out, abs, ejected)
		if copyUsesRT {
			copiedRT = true
		}
		rel, err := filepath.Rel(modRoot, owner.Dir)
		if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
			return false, fmt.Errorf("owned request for %q outside its module", req.Owner)
		}
		dstDir := filepath.Join(dstRoot, rel)
		if err := ownedCheckDir(dstRoot, dstDir); err != nil {
			return false, fmt.Errorf("owned request for %q: %v", req.Owner, err)
		}
		dests = append(dests, &ownedDest{req: req, owner: owner, dstRoot: dstRoot, dstDir: dstDir})
	}

	dests, err = applyOwnedConditionsAndCollisions(ctx, plan, dests)
	if err != nil {
		return false, err
	}

	usedRT := copiedRT
	for _, d := range dests {
		if _, err := os.Lstat(d.dst); err == nil {
			return false, fmt.Errorf("owned output %q already exists (conflicting generated declarations)", d.dst)
		} else if !os.IsNotExist(err) {
			return false, fmt.Errorf("stat owned output %q: %v", d.dst, err)
		}
		if d.usedRT {
			usedRT = true
		}
		if err := os.MkdirAll(d.dstDir, 0777); err != nil {
			return false, fmt.Errorf("make owned directory: %v", err)
		}
		if fi, err := os.Lstat(d.dstDir); err != nil || fi.Mode()&os.ModeSymlink != 0 {
			return false, fmt.Errorf("owned directory %q is not a directory", d.dstDir)
		}
		if err := os.WriteFile(d.dst, d.fixed, 0666); err != nil {
			return false, fmt.Errorf("write owned helper: %v", err)
		}
		queueEjectGofmt(d.dst)
	}
	return usedRT, nil
}

type ownedDest struct {
	req     *logosowned.Request
	owner   *load.Package
	dstRoot string
	dstDir  string
	name    string
	dst     string
	fixed   []byte
	usedRT  bool
}

func gatherOwnedRequests(staging string) ([]logosowned.Request, error) {
	var sidecars []string
	err := filepath.WalkDir(staging, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && d.Name() == logosowned.FileName {
			sidecars = append(sidecars, path)
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("gather owned sidecars: %v", err)
	}
	slices.Sort(sidecars)
	var all []logosowned.Request
	for _, sidecar := range sidecars {
		data, err := os.ReadFile(sidecar)
		if err != nil {
			return nil, fmt.Errorf("read owned sidecar: %v", err)
		}
		reqs, err := logosowned.Decode(data)
		if err != nil {
			return nil, fmt.Errorf("decode owned sidecar %s: %v", sidecar, err)
		}
		all = append(all, reqs...)
	}
	return dedupeOwnedRequests(all)
}

func dedupeOwnedRequests(reqs []logosowned.Request) ([]logosowned.Request, error) {
	seen := map[string]logosowned.Request{}
	for _, r := range reqs {
		key := r.Owner + "\x00" + r.ID + "\x00" + ownedDeclarationsKey(r.Declarations)
		if prev, ok := seen[key]; ok {
			if !ownedRequestsEqual(prev, r) {
				return nil, fmt.Errorf("conflicting owned requests for owner %q id %s", r.Owner, r.ID)
			}
			continue
		}
		seen[key] = r
	}
	out := make([]logosowned.Request, 0, len(seen))
	for _, r := range seen {
		out = append(out, r)
	}
	slices.SortFunc(out, func(a, b logosowned.Request) int {
		if c := strings.Compare(a.Owner, b.Owner); c != 0 {
			return c
		}
		if c := strings.Compare(a.Package, b.Package); c != 0 {
			return c
		}
		if c := strings.Compare(a.ID, b.ID); c != 0 {
			return c
		}
		return strings.Compare(ownedDeclarationsKey(a.Declarations), ownedDeclarationsKey(b.Declarations))
	})
	return out, nil
}

func ownedDeclarationsKey(decls []logosowned.Declaration) string {
	var b strings.Builder
	for _, d := range decls {
		b.WriteString(d.Owner)
		b.WriteByte(0)
		b.WriteString(d.File)
		b.WriteByte(1)
	}
	return b.String()
}

func ownedRequestsEqual(a, b logosowned.Request) bool {
	if a.Owner != b.Owner || a.Package != b.Package || a.ID != b.ID || a.Source != b.Source {
		return false
	}
	if len(a.Declarations) != len(b.Declarations) {
		return false
	}
	for i := range a.Declarations {
		if a.Declarations[i] != b.Declarations[i] {
			return false
		}
	}
	return true
}

func selectOwnedPackage(pkgs []*load.Package, owner, name string) *load.Package {
	var fallback *load.Package
	for _, p := range pkgs {
		if p.ImportPath != owner || name != "" && p.Name != name {
			continue
		}
		if p.ForTest == "" && p.Internal.TestmainGo == nil {
			return p
		}
		if fallback == nil {
			fallback = p
		}
	}
	return fallback
}

func findOwnedOwner(plan *ejectPlan, owner string) *load.Package {
	all := slices.Clone(plan.main.pkgs)
	for _, path := range slices.Sorted(maps.Keys(plan.depByPath)) {
		all = append(all, plan.depByPath[path].pkgs...)
	}
	return selectOwnedPackage(all, owner, "")
}

func ownedFindDeclPackage(plan *ejectPlan, owner string) *load.Package {
	if plan != nil && plan.provenance != nil {
		if pkgs, ok := plan.provenance[owner]; ok && len(pkgs) > 0 {
			if hit := selectOwnedPackage(pkgs, owner, ""); hit != nil {
				return hit
			}
		}
	}
	return findOwnedOwner(plan, owner)
}

func ownedOutputRoots(plan *ejectPlan, owner *load.Package, out, abs string, ejected map[string]string) (dstRoot, modRoot string, usesRT bool) {
	if owner.Module.Main {
		return out, plan.main.dir, false
	}
	m := plan.depByPath[owner.Module.Path]
	if m == nil {
		base.Fatalf("logos eject: owned request for %q has no module", owner.ImportPath)
	}
	if dst, ok := ejected[m.path]; ok {
		return dst, m.dir, false
	}
	dst := filepath.Join(out, "_logos_deps", m.path+"@"+m.version)
	ejected[m.path] = dst
	usesRT = copyModuleTree(m.dir, dst, ejectSkipDir(m.dir, abs, out))
	plan.deps = append(plan.deps, m)
	slices.SortFunc(plan.deps, func(a, b *ejectModule) int { return strings.Compare(a.path, b.path) })
	return dst, m.dir, usesRT
}

func ownedCheckDir(dstRoot, dst string) error {
	rel, err := filepath.Rel(dstRoot, dst)
	if err != nil {
		return fmt.Errorf("resolve %q: %v", dst, err)
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return fmt.Errorf("%q escapes its package", dst)
	}
	if rel == "." {
		return ownedCheckSymlink(dstRoot)
	}
	cur := dstRoot
	for part := range strings.SplitSeq(rel, string(filepath.Separator)) {
		cur = filepath.Join(cur, part)
		if err := ownedCheckSymlink(cur); err != nil {
			if os.IsNotExist(err) {
				return nil
			}
			return err
		}
	}
	return nil
}

func ownedCheckSymlink(path string) error {
	fi, err := os.Lstat(path)
	if err != nil {
		return err
	}
	if fi.Mode()&os.ModeSymlink != 0 {
		return fmt.Errorf("%q is a symlink", path)
	}
	if !fi.IsDir() && filepath.Ext(path) == "" {
		return fmt.Errorf("%q is not a directory", path)
	}
	return nil
}

func gorootContains(dir string) bool {
	goroot := cfg.GOROOT
	if goroot == "" || dir == "" {
		return false
	}
	rel, err := filepath.Rel(goroot, dir)
	if err != nil {
		return false
	}
	return rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

func applyOwnedConditionsAndCollisions(ctx context.Context, plan *ejectPlan, dests []*ownedDest) ([]*ownedDest, error) {
	type packageKey struct{ dir, name string }
	groups := map[packageKey][]*ownedDest{}
	var keys []packageKey
	for _, d := range dests {
		key := packageKey{d.dstDir, d.owner.Name}
		if groups[key] == nil {
			keys = append(keys, key)
		}
		groups[key] = append(groups[key], d)
	}
	slices.SortFunc(keys, func(a, b packageKey) int {
		if c := strings.Compare(a.dir, b.dir); c != 0 {
			return c
		}
		return strings.Compare(a.name, b.name)
	})
	writers := make([]*ownedDest, 0, len(dests))
	for _, key := range keys {
		groupWriters, err := ownedResolveGroup(ctx, plan, key.dir, groups[key])
		if err != nil {
			return nil, fmt.Errorf("owned request for %q: %v", groups[key][0].req.Owner, err)
		}
		writers = append(writers, groupWriters...)
	}
	return writers, nil
}

func ownedResolveGroup(ctx context.Context, plan *ejectPlan, dir string, dests []*ownedDest) ([]*ownedDest, error) {
	pkgName := dests[0].owner.Name

	srcDecls := map[string]string{}
	entries, err := os.ReadDir(dir)
	if err != nil && !os.IsNotExist(err) {
		return nil, fmt.Errorf("read owner directory: %v", err)
	}
	fset := token.NewFileSet()
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".go") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			return nil, fmt.Errorf("read owner file %s: %v", e.Name(), err)
		}
		f, err := parser.ParseFile(fset, e.Name(), data, parser.ParseComments)
		if err != nil {
			return nil, fmt.Errorf("parse owner file %s: %v", e.Name(), err)
		}
		if f.Name.Name != pkgName {
			continue
		}
		for _, name := range ownedFileDeclNames(f) {
			if name == "_" || name == "init" || strings.Contains(name, ".") {
				continue
			}
			if prev, ok := srcDecls[name]; ok {
				_ = prev
				continue
			}
			srcDecls[name] = e.Name()
		}
	}

	for _, d := range dests {
		if _, err := ownedDeclNames(d.req.Owner, []byte(d.req.Source)); err != nil {
			return nil, err
		}
	}

	variants, err := ownedGroupVariants(dests)
	if err != nil {
		return nil, err
	}
	merged := map[string]string{}
	for _, group := range variants {
		names, err := ownedDeclNames(group[0].req.Owner, []byte(group[0].req.Source))
		if err != nil {
			return nil, err
		}
		for _, name := range names {
			if name == "_" || name == "init" || strings.Contains(name, ".") {
				continue
			}
			if prev, ok := merged[name]; ok && prev != group[0].req.ID {
				return nil, fmt.Errorf("owned helpers for %q both declare %q (%s and %s)", group[0].req.Owner, name, prev, group[0].req.ID)
			}
			merged[name] = group[0].req.ID
		}
	}

	for _, group := range variants {
		names, err := ownedDeclNames(group[0].req.Owner, []byte(group[0].req.Source))
		if err != nil {
			return nil, err
		}
		for _, decl := range names {
			if file, ok := srcDecls[decl]; ok {
				return nil, fmt.Errorf("owned helper for %q declares %q, already declared in %s", group[0].req.Owner, decl, file)
			}
		}
	}

	var writers []*ownedDest
	for _, group := range variants {
		slices.SortFunc(group, func(a, b *ownedDest) int {
			if c := strings.Compare(a.req.ID, b.req.ID); c != 0 {
				return c
			}
			return strings.Compare(ownedDeclarationsKey(a.req.Declarations), ownedDeclarationsKey(b.req.Declarations))
		})
		var prodDests []*ownedDest
		var testDests []*ownedDest
		var prodEffs []constraint.Expr
		var testEffs []constraint.Expr
		for _, v := range group {
			eff, test, err := ownedEffectiveCondition(plan, v.req)
			if err != nil {
				return nil, err
			}
			if test {
				testDests = append(testDests, v)
				testEffs = append(testEffs, eff)
			} else {
				prodDests = append(prodDests, v)
				prodEffs = append(prodEffs, eff)
			}
		}
		body, err := ownedStripHeader(group[0].req.Source)
		if err != nil {
			return nil, fmt.Errorf("helper %s: %v", group[0].req.ID, err)
		}
		emitOne := func(rep *ownedDest, cond constraint.Expr, test bool) error {
			header := ownedRenderHeader(cond)
			finalSource := header + body
			fixed, uses, ok := remapEjectedFile([]byte(finalSource), "")
			if !ok {
				return fmt.Errorf("helper %s does not parse after condition merge", rep.req.ID)
			}
			fixed = gofmtRun(ctx, []string{gofmtBinary()}, fixed)
			sum := sha256.Sum256(fixed)
			hash := hex.EncodeToString(sum[:])
			if test {
				rep.name = "logos_owned_" + hash + "_test.go"
			} else {
				rep.name = "logos_owned_" + hash + ".go"
			}
			rep.dst = filepath.Join(rep.dstDir, rep.name)
			rep.fixed = fixed
			rep.usedRT = uses
			writers = append(writers, rep)
			return nil
		}
		if len(prodDests) == 0 {
			final := ownedOrExprs(testEffs)
			if err := emitOne(group[0], final, true); err != nil {
				return nil, err
			}
			continue
		}
		prodCond := ownedOrExprs(prodEffs)
		if err := emitOne(prodDests[0], prodCond, false); err != nil {
			return nil, err
		}
		if len(testDests) == 0 {
			continue
		}
		if prodCond == nil {
			continue
		}
		testOr := ownedOrExprs(testEffs)
		if testOr != nil && testOr.String() == prodCond.String() {
			continue
		}
		notProd := &constraint.NotExpr{X: prodCond}
		testCond := ownedCombineAnd([]constraint.Expr{testOr, notProd})
		if err := emitOne(testDests[0], testCond, true); err != nil {
			return nil, err
		}
	}
	slices.SortFunc(writers, func(a, b *ownedDest) int {
		return strings.Compare(a.dst, b.dst)
	})
	return writers, nil
}

func ownedEffectiveCondition(plan *ejectPlan, req *logosowned.Request) (constraint.Expr, bool, error) {
	consumer, err := ownedHeaderExprOf(req.Source)
	if err != nil {
		return nil, false, fmt.Errorf("helper %s: %v", req.ID, err)
	}
	conds := []constraint.Expr{consumer}
	isTest := false
	for _, decl := range req.Declarations {
		cond, test, err := ownedDeclarationCondition(plan, decl.Owner, decl.File)
		if err != nil {
			return nil, false, fmt.Errorf("helper %s declaration %s:%s: %v", req.ID, decl.Owner, decl.File, err)
		}
		conds = append(conds, cond)
		if test {
			isTest = true
		}
	}
	return ownedCombineAnd(conds), isTest, nil
}

func ownedDeclarationCondition(plan *ejectPlan, declOwner, declFile string) (constraint.Expr, bool, error) {
	pkg := ownedFindDeclPackage(plan, declOwner)
	if pkg == nil {
		return nil, false, fmt.Errorf("declaration owner %q not in package plan", declOwner)
	}
	original, isTest, err := ownedSelectedFile(pkg, declFile)
	if err != nil {
		// Inactive helpers retain actual loader-selected inventories. A
		// declaration ignored by the host may be selected by one of them;
		// filenames absent from every loaded inventory still fail below.
		for _, candidate := range plan.provenance[declOwner] {
			if selected, test, selectErr := ownedSelectedFile(candidate, declFile); selectErr == nil {
				pkg, original, isTest, err = candidate, selected, test, nil
				break
			}
		}
	}
	if err != nil {
		return nil, false, err
	}
	data, err := os.ReadFile(filepath.Join(pkg.Dir, original))
	if err != nil {
		return nil, false, fmt.Errorf("read declaration file %s of %q: %v", original, declOwner, err)
	}
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, original, data, parser.ImportsOnly|parser.ParseComments|parser.SkipObjectResolution)
	if err != nil {
		return nil, false, fmt.Errorf("parse declaration file %s of %q: %v", original, declOwner, err)
	}
	if f.Name.Name != pkg.Name {
		return nil, false, fmt.Errorf("declaration file %s of %q has package %q, want %q", original, declOwner, f.Name.Name, pkg.Name)
	}
	cond, err := ownedFileCondition(original, f)
	if err != nil {
		return nil, false, fmt.Errorf("condition of %s: %v", original, err)
	}
	return cond, isTest, nil
}

func ownedSelectedFile(pkg *load.Package, declFile string) (string, bool, error) {
	if len(pkg.GoFiles) == 0 && len(pkg.CgoFiles) == 0 && len(pkg.TestGoFiles) == 0 && len(pkg.CompiledGoFiles) == 0 && len(pkg.IgnoredGoFiles) == 0 {
		return "", false, fmt.Errorf("owner %q has no source inventory (no all-files scan)", pkg.ImportPath)
	}
	for _, name := range pkg.CompiledGoFiles {
		if name == declFile {
			return "", false, fmt.Errorf("declaration file %q of %q is a generated cgo output with no source mapping (GoFiles=%q CgoFiles=%q CompiledGoFiles=%q)", declFile, pkg.ImportPath, pkg.GoFiles, pkg.CgoFiles, pkg.CompiledGoFiles)
		}
	}
	for _, name := range pkg.IgnoredGoFiles {
		if name == declFile {
			return "", false, fmt.Errorf("declaration file %q of %q is not selected (ignored)", declFile, pkg.ImportPath)
		}
	}
	for _, name := range pkg.GoFiles {
		if name == declFile {
			return name, false, nil
		}
	}
	for _, name := range pkg.CgoFiles {
		if name == declFile {
			return name, false, nil
		}
	}
	for _, name := range pkg.TestGoFiles {
		if name == declFile {
			return name, true, nil
		}
	}
	for _, name := range pkg.GoFiles {
		if logosfile.Is(name) && logosStem(name)+".go" == declFile {
			return name, false, nil
		}
	}
	for _, name := range pkg.TestGoFiles {
		if logosfile.Is(name) && logosStem(name)+".go" == declFile {
			return name, true, nil
		}
	}
	return "", false, fmt.Errorf("declaration file %q not in selected sources of %q", declFile, pkg.ImportPath)
}

func ownedGroupVariants(dests []*ownedDest) ([][]*ownedDest, error) {
	byBody := map[string][]*ownedDest{}
	var bodies []string
	for _, d := range dests {
		body, err := ownedStripHeader(d.req.Source)
		if err != nil {
			return nil, fmt.Errorf("helper %s: %v", d.req.ID, err)
		}
		if byBody[body] == nil {
			bodies = append(bodies, body)
		}
		byBody[body] = append(byBody[body], d)
	}
	slices.Sort(bodies)
	groups := make([][]*ownedDest, 0, len(bodies))
	for _, body := range bodies {
		groups = append(groups, byBody[body])
	}
	return groups, nil
}

func ownedStripHeader(source string) (string, error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", []byte(source), parser.PackageClauseOnly)
	if err != nil {
		return "", err
	}
	pkgOffset := fset.Position(f.Package).Offset
	var kept []string
	for _, ln := range strings.Split(source[:pkgOffset], "\n") {
		t := strings.TrimSpace(ln)
		if t == "" || constraint.IsGoBuild(t) || constraint.IsPlusBuild(t) {
			continue
		}
		kept = append(kept, ln)
	}
	if len(kept) == 0 {
		return source[pkgOffset:], nil
	}
	return strings.Join(kept, "\n") + "\n" + source[pkgOffset:], nil
}

func ownedHeaderExprOf(source string) (constraint.Expr, error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "owned.go", []byte(source), parser.ParseComments)
	if err != nil {
		return nil, err
	}
	return ownedHeaderExpr(f)
}

var ownedGoosTags = []string{"aix", "android", "darwin", "dragonfly", "freebsd", "hurd", "illumos", "ios", "js", "linux", "nacl", "netbsd", "openbsd", "plan9", "solaris", "windows", "zos"}

var ownedGoarchTags = []string{"386", "amd64", "arm", "arm64", "loong64", "mips", "mips64", "mips64le", "mipsle", "ppc64", "ppc64le", "riscv", "riscv64", "s390x", "sparc64", "wasm"}

func ownedFileCondition(name string, f *ast.File) (constraint.Expr, error) {
	var parts []constraint.Expr
	h, err := ownedHeaderExpr(f)
	if err != nil {
		return nil, err
	}
	if h != nil {
		parts = append(parts, h)
	}
	for _, tag := range ownedFilenameTags(name) {
		parts = append(parts, &constraint.TagExpr{Tag: tag})
	}
	if ownedHasCgo(f) {
		parts = append(parts, &constraint.TagExpr{Tag: "cgo"})
	}
	return ownedCombineAnd(parts), nil
}

func ownedHeaderExpr(f *ast.File) (constraint.Expr, error) {
	var modern, legacy constraint.Expr
	for _, cg := range f.Comments {
		if cg.Pos() > f.Pos() {
			break
		}
		for _, c := range cg.List {
			switch {
			case constraint.IsGoBuild(c.Text):
				expr, err := constraint.Parse(c.Text)
				if err != nil {
					return nil, err
				}
				modern = expr
			case constraint.IsPlusBuild(c.Text):
				expr, err := constraint.Parse(c.Text)
				if err != nil {
					return nil, err
				}
				if legacy == nil {
					legacy = expr
				} else {
					legacy = &constraint.AndExpr{X: legacy, Y: expr}
				}
			}
		}
	}
	if modern != nil {
		return modern, nil
	}
	return legacy, nil
}

func ownedFilenameTags(name string) []string {
	var base string
	if logosfile.Is(name) {
		base = logosStem(name)
	} else {
		base = strings.TrimSuffix(name, ".go")
	}
	base = strings.TrimSuffix(base, "_test")
	for _, goarch := range ownedGoarchTags {
		if !strings.HasSuffix(base, "_"+goarch) {
			continue
		}
		for _, goos := range ownedGoosTags {
			if strings.HasSuffix(base, "_"+goos+"_"+goarch) {
				return []string{goos, goarch}
			}
		}
		return []string{goarch}
	}
	for _, goos := range ownedGoosTags {
		if strings.HasSuffix(base, "_"+goos) {
			return []string{goos}
		}
	}
	return nil
}

func ownedHasCgo(f *ast.File) bool {
	for _, imp := range f.Imports {
		if path, err := strconv.Unquote(imp.Path.Value); err == nil && path == "C" {
			return true
		}
	}
	return false
}

func ownedOrExprs(conds []constraint.Expr) constraint.Expr {
	for _, c := range conds {
		if c == nil {
			return nil
		}
	}
	seen := map[string]bool{}
	var uniq []constraint.Expr
	for _, c := range conds {
		if c == nil {
			continue
		}
		if s := c.String(); seen[s] {
			continue
		} else {
			seen[s] = true
		}
		uniq = append(uniq, c)
	}
	var out constraint.Expr
	for _, c := range uniq {
		if out == nil {
			out = c
			continue
		}
		out = &constraint.OrExpr{X: out, Y: c}
	}
	return out
}

func ownedCombineAnd(conds []constraint.Expr) constraint.Expr {
	seen := map[string]bool{}
	var uniq []constraint.Expr
	for _, c := range conds {
		if c == nil {
			continue
		}
		if s := c.String(); seen[s] {
			continue
		} else {
			seen[s] = true
		}
		uniq = append(uniq, c)
	}
	var out constraint.Expr
	for _, c := range uniq {
		if out == nil {
			out = c
			continue
		}
		out = &constraint.AndExpr{X: out, Y: c}
	}
	return out
}

func ownedRenderHeader(expr constraint.Expr) string {
	if expr == nil {
		return ""
	}
	header := "//go:build " + expr.String() + "\n"
	if lines, err := constraint.PlusBuildLines(expr); err == nil {
		header += strings.Join(lines, "\n") + "\n"
	}
	return header + "\n"
}

func ownedDeclNames(owner string, data []byte) ([]string, error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "owned.go", data, parser.SkipObjectResolution)
	if err != nil {
		return nil, fmt.Errorf("parse generated helper: %v", err)
	}
	var names []string
	seen := map[string]bool{}
	bump := func(name string) error {
		if name == "_" || name == "init" {
			return nil
		}
		if seen[name] {
			return fmt.Errorf("owned helper for %q declares %q twice", owner, name)
		}
		seen[name] = true
		names = append(names, name)
		return nil
	}
	for _, decl := range f.Decls {
		switch d := decl.(type) {
		case *ast.FuncDecl:
			if d.Recv == nil {
				if err := bump(d.Name.Name); err != nil {
					return nil, err
				}
				continue
			}
			if err := bump(ownedReceiver(d.Recv.List) + "." + d.Name.Name); err != nil {
				return nil, err
			}
		case *ast.GenDecl:
			for _, spec := range d.Specs {
				switch s := spec.(type) {
				case *ast.TypeSpec:
					if err := bump(s.Name.Name); err != nil {
						return nil, err
					}
				case *ast.ValueSpec:
					for _, name := range s.Names {
						if err := bump(name.Name); err != nil {
							return nil, err
						}
					}
				}
			}
		}
	}
	return names, nil
}

func ownedFileDeclNames(f *ast.File) []string {
	var names []string
	for _, decl := range f.Decls {
		switch d := decl.(type) {
		case *ast.FuncDecl:
			if d.Recv == nil {
				names = append(names, d.Name.Name)
				continue
			}
			names = append(names, ownedReceiver(d.Recv.List)+"."+d.Name.Name)
		case *ast.GenDecl:
			for _, spec := range d.Specs {
				switch s := spec.(type) {
				case *ast.TypeSpec:
					names = append(names, s.Name.Name)
				case *ast.ValueSpec:
					for _, name := range s.Names {
						names = append(names, name.Name)
					}
				}
			}
		}
	}
	return names
}

func ownedReceiver(fields []*ast.Field) string {
	if len(fields) == 0 {
		return ""
	}
	t := fields[0].Type
	for {
		switch v := t.(type) {
		case *ast.StarExpr:
			t = v.X
		case *ast.IndexExpr:
			t = v.X
		case *ast.IndexListExpr:
			t = v.X
		default:
			return ownedTypeName(t)
		}
	}
}

func ownedTypeName(e ast.Expr) string {
	switch v := e.(type) {
	case *ast.Ident:
		return v.Name
	case *ast.SelectorExpr:
		return ownedTypeName(v.X) + "." + v.Sel.Name
	default:
		return ""
	}
}
