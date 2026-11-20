// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sgutil

import (
	"fmt"
	"go/ast"
	"go/build"
	"go/build/constraint"
	"go/token"
	"io"
	"path/filepath"
	"slices"
	"strings"
)

// stableArchsimdGOARCHes lists the architectures whose archsimd API can be
// generally available. The archsimd API for any other architecture is
// experimental until its architecture is added here.
var stableArchsimdGOARCHes = []string{"amd64", "arm64", "wasm"}

// maxBuildTags limits the number of distinct build tags in a //go:build line
// that [ParseBuildConstraints] has to try every assignment of.
const maxBuildTags = 16

// BuildConstraints are the build constraints of a SIMD source file: its
// //go:build line, and the GOARCH that its file name implies, if any.
type BuildConstraints struct {
	expr     constraint.Expr // the //go:build constraint, or nil
	nameArch string          // the GOARCH in the file name, or "" if there's no GOARCH in the file name or a GOARCH that isn't in stableArchsimdGOARCHes

	phase          Phase
	stableArchOnly bool

	free []string // tags whose assignments we had to try (for testing)
}

// ParseBuildConstraints returns the build constraints of file, which must have
// been parsed with comments using fset. It's an error if the file never
// builds, or if its //go:build line is too complicated to analyze.
func ParseBuildConstraints(fset *token.FileSet, file *ast.File) (*BuildConstraints, error) {
	expr, err := buildConstraint(file)
	if err != nil {
		return nil, err
	}
	c := &BuildConstraints{
		expr:     expr,
		nameArch: nameGOARCH(filepath.Base(fset.File(file.Pos()).Name())),
	}

	// We work out the phase and whether the file is only for stable archsimd
	// GOARCHes without knowing which build tags are GOARCHes, or which
	// GOARCHes exist. Every tag other than goexperiment.simd and
	// goexperiment.simdstable is unknown, even the stable archsimd GOARCH
	// tags, and we try every assignment of true or false to the unknown tags
	// with each GOEXPERIMENT setting: neither experiment, simdstable alone,
	// and simd+simdstable. We check every satisfying assignment to see whether
	// it's satisfied even with no stable archsimd GOARCH set. For example,
	// "goexperiment.simdstable && (amd64 || mips)" builds with simdstable
	// alone, so it's PhaseStable, and it builds with mips set and no stable
	// archsimd GOARCH, so it isn't only for stable archsimd GOARCHes.
	//
	// Trying every assignment is exact, except that some assignments can't
	// happen in a real build, such as one that sets two GOARCHes. Those can
	// only make us report that a file isn't only for stable archsimd GOARCHes
	// when it is, never the reverse, and they don't arise from realistic
	// constraints.
	//
	// A file name that restricts the file to one stable archsimd GOARCH, as
	// ops_amd64.go or slicepart_wasm_test.go does, is like an additional
	// conjunct with that GOARCH. Other file names don't matter: a file named
	// ops_riscv64.go just looks like a file that isn't restricted to stable
	// archsimd GOARCHes.
	full := expr
	if c.nameArch != "" {
		tag := &constraint.TagExpr{Tag: c.nameArch}
		if full == nil {
			full = tag
		} else {
			full = &constraint.AndExpr{X: full, Y: tag}
		}
	}
	if full == nil {
		// The file always builds, for every GOARCH and in any phase.
		return c, nil
	}

	// A tag reachable from the root only through &&s must be set for the file
	// to build, and a !tag reachable that way must be unset (and likewise
	// through the ||s of a negated expression, by De Morgan's laws). Fix
	// those, so we only have to try assignments of the other ("free") tags.
	// This leaves nothing to try for most real constraints, such as
	// "goexperiment.simdstable && amd64". sense is whether x must be true.
	fixed := make(map[string]bool)
	conflict := false
	var conjuncts func(x constraint.Expr, sense bool)
	conjuncts = func(x constraint.Expr, sense bool) {
		switch x := x.(type) {
		case *constraint.AndExpr:
			if sense {
				conjuncts(x.X, sense)
				conjuncts(x.Y, sense)
			}
		case *constraint.OrExpr:
			if !sense {
				conjuncts(x.X, sense)
				conjuncts(x.Y, sense)
			}
		case *constraint.NotExpr:
			conjuncts(x.X, !sense)
		case *constraint.TagExpr:
			if v, ok := fixed[x.Tag]; ok && v != sense {
				// The tag must be both set and unset.
				conflict = true
			}
			fixed[x.Tag] = sense
		}
	}
	conjuncts(full, true)
	if conflict {
		return nil, fmt.Errorf("build constraint %s never holds", full)
	}

	// Number the free tags: those that aren't fixed, other than the SIMD
	// GOEXPERIMENT tags, whose values depend on the phase we try.
	free := make(map[string]int)
	var collect func(x constraint.Expr)
	collect = func(x constraint.Expr) {
		switch x := x.(type) {
		case *constraint.TagExpr:
			_, isFixed := fixed[x.Tag]
			_, seen := free[x.Tag]
			if !isFixed && !seen && !slices.Contains(AllPhaseTags, x.Tag) {
				free[x.Tag] = len(free)
			}
		case *constraint.NotExpr:
			collect(x.X)
		case *constraint.AndExpr:
			collect(x.X)
			collect(x.Y)
		case *constraint.OrExpr:
			collect(x.X)
			collect(x.Y)
		default:
			panic(fmt.Sprintf("unexpected constraint expression %T", x))
		}
	}
	collect(full)
	if len(free) > maxBuildTags {
		return nil, fmt.Errorf("build constraint %s has too many tags to check; simplify the //go:build line", expr)
	}
	// Record free tags for testing.
	for tag := range free {
		c.free = append(c.free, tag)
	}

	// try reports whether the file builds with the SIMD GOEXPERIMENT tags in
	// expTags set (and the others unset), and if so, whether it only builds
	// when a stable archsimd GOARCH is set.
	try := func(expTags []string) (builds, stableArchOnly bool) {
		stableArchOnly = true
		// Enumerate all possible assignments of free tags.
		for bits := 0; bits < 1<<len(free); bits++ {
			isSet := func(tag string) bool {
				// Is it a SIMD phase tag?
				if slices.Contains(AllPhaseTags, tag) {
					return slices.Contains(expTags, tag)
				}
				if v, ok := fixed[tag]; ok {
					return v
				}
				// Every tag in the expression is fixed or free, so this
				// is only !ok for tags that aren't in the expression at
				// all, such as stable archsimd GOARCHes it doesn't
				// mention. Those are unset.
				i, ok := free[tag]
				return ok && bits&(1<<i) != 0
			}
			if !full.Eval(isSet) {
				continue
			}
			// We found some configuration that builds.
			builds = true
			if !slices.ContainsFunc(stableArchsimdGOARCHes, isSet) {
				// We found a configuration that builds even if we're
				// not on a stable GOARCH.
				stableArchOnly = false
				break
			}
		}
		return builds, stableArchOnly
	}

	// The phase is the first, from least to most experimental, in which the
	// file builds.
	for _, phase := range []Phase{0, PhaseStable, PhaseExperiment} {
		builds, stableArchOnly := try(phase.BuildTags())
		if builds {
			c.phase, c.stableArchOnly = phase, stableArchOnly
			return c, nil
		}
	}
	return nil, fmt.Errorf("build constraint %s never holds", full)
}

// Phase returns the phase whose GOEXPERIMENT the file needs: [PhaseStable] if
// it builds with GOEXPERIMENT=simdstable, [PhaseExperiment] if it builds only
// with GOEXPERIMENT=simd (which implies simdstable), or 0 if it builds with
// neither.
func (c *BuildConstraints) Phase() Phase { return c.phase }

// StableArchOnly reports whether, with the GOEXPERIMENT of its phase, the file
// only builds for architectures whose archsimd API can be generally
// available (amd64, arm64 and wasm). Only such a file may declare generally
// available archsimd API.
func (c *BuildConstraints) StableArchOnly() bool { return c.stableArchOnly }

// buildConstraint returns file's //go:build constraint, or nil if it has none.
func buildConstraint(file *ast.File) (constraint.Expr, error) {
	// The //go:build line must come before the package clause.
	for _, cg := range file.Comments {
		if cg.Pos() > file.Package {
			break
		}
		for _, c := range cg.List {
			if constraint.IsGoBuild(c.Text) {
				return constraint.Parse(c.Text)
			}
		}
	}
	return nil, nil
}

// nameGOARCH returns the stable archsimd GOARCH that the file name filename
// restricts a file to, or "" if it doesn't restrict the GOARCH or restricts it
// to an unknown GOARCH.
func nameGOARCH(filename string) string {
	// Internally, go/build knows how to parse any GOARCH out of a file name,
	// but it doesn't expose that functionality directly. Instead, ask it
	// whether the name matches each stable archsimd GOARCH, one at a time.
	var match []string
	for _, arch := range stableArchsimdGOARCHes {
		// Use the same GOOS for every GOARCH, so that a GOOS in the name
		// doesn't make it match some of them and not others. This happens
		// to work for wasm, too, even though linux/wasm isn't a valid
		// platform.
		ctxt := build.Context{GOARCH: arch, GOOS: "linux", Compiler: "gc"}
		// We process the go:build line ourselves, so give go/build an empty
		// source file.
		ctxt.OpenFile = func(string) (io.ReadCloser, error) {
			return io.NopCloser(strings.NewReader("package p\n")), nil
		}
		if ok, err := ctxt.MatchFile(".", filename); err == nil && ok {
			match = append(match, arch)
		}
	}
	switch len(match) {
	case 0:
		// Doesn't build on any stable GOARCH.
		return ""
	case 1:
		return match[0]
	case len(stableArchsimdGOARCHes):
		// The name doesn't restrict the GOARCH.
		return ""
	}
	panic(fmt.Sprintf("file name %s matches more than one GOARCH: %v", filename, match))
}
