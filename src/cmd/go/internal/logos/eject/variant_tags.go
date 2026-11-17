// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eject

import (
	"fmt"
	"go/build"
	"go/build/constraint"
	"internal/syslist"
	"path/filepath"
	"slices"

	"cmd/go/internal/base"
)

// matchVariantTags solves the parsed constraint before asking MatchFile to
// validate a complete context. customUniq is the caller's sorted list of tags
// that may change; unrelated host tags and all other tag facts stay fixed.
func matchVariantTags(pkgDir, name string, expr constraint.Expr, ctxt build.Context, hostTags, customUniq []string) ([]string, bool) {
	goos, goarch := ctxt.GOOS, ctxt.GOARCH
	values := map[string]bool{goos: true, goarch: true, ctxt.Compiler: true, "cgo": ctxt.CgoEnabled}
	for _, tags := range [][]string{hostTags, ctxt.ToolTags, ctxt.ReleaseTags} {
		for _, tag := range tags {
			values[tag] = true
		}
	}
	// These aliases have the same truth values as go/build.Context.matchTag.
	switch goos {
	case "android":
		values["linux"] = true
	case "illumos":
		values["solaris"] = true
	case "ios":
		values["darwin"] = true
	}
	values["unix"] = values["unix"] || syslist.UnixOS[goos]
	values["boringcrypto"] = values["goexperiment.boringcrypto"]
	var mentioned []string
	if expr != nil {
		collectConstraintTags(expr, &mentioned)
	}
	for _, tag := range mentioned {
		// Record absent fixed tags as false. Only an unassigned custom tag
		// is absent from values when the solver starts.
		if _, assigned := values[tag]; !assigned {
			values[tag] = false
		}
	}
	var selected []string
	match := func() bool {
		tags := make([]string, 0, len(hostTags)+len(customUniq))
		for _, tag := range hostTags {
			_, mutable := slices.BinarySearch(customUniq, tag)
			if value, assigned := values[tag]; !mutable || !assigned || value {
				tags = append(tags, tag)
			}
		}
		for _, tag := range customUniq {
			if values[tag] {
				tags = append(tags, tag)
			}
		}
		slices.Sort(tags)
		tags = slices.Compact(tags)
		ctxt.BuildTags = tags
		ok, err := ctxt.MatchFile(pkgDir, name)
		if err != nil {
			base.Fatalf("logos eject: match inactive file %s: %v", filepath.Join(pkgDir, name), err)
		}
		if ok {
			selected = tags
		}
		return ok
	}
	// Preserve an already satisfying host assignment. Adding a redundant tag
	// can select different dependency files even when this file still matches.
	if expr == nil || expr.Eval(func(tag string) bool { return values[tag] }) {
		ok := match()
		return selected, ok
	}
	for _, tag := range customUniq {
		delete(values, tag)
	}
	ok := satisfyVariantTags(expr, true, values, match)
	return selected, ok
}

// satisfyVariantTags visits satisfying assignments without enumerating a
// machine-word mask. An absent value is an unassigned custom tag. Failed
// branches restore values; successful branches retain their witness. The
// continuation lets an earlier choice backtrack when a later conjunct fails.
// Forced conjunctions take linear solver work; general formulas can require
// exponential search, as Boolean satisfiability does.
func satisfyVariantTags(expr constraint.Expr, want bool, values map[string]bool, next func() bool) bool {
	switch e := expr.(type) {
	case *constraint.TagExpr:
		if value, assigned := values[e.Tag]; assigned {
			return value == want && next()
		}
		values[e.Tag] = want
		if next() {
			return true
		}
		delete(values, e.Tag)
		return false
	case *constraint.NotExpr:
		return satisfyVariantTags(e.X, !want, values, next)
	case *constraint.AndExpr:
		if want {
			return satisfyVariantTags(e.X, true, values, func() bool {
				return satisfyVariantTags(e.Y, true, values, next)
			})
		}
		return satisfyVariantTags(e.X, false, values, next) || satisfyVariantTags(e.Y, false, values, next)
	case *constraint.OrExpr:
		if !want {
			return satisfyVariantTags(e.X, false, values, func() bool {
				return satisfyVariantTags(e.Y, false, values, next)
			})
		}
		return satisfyVariantTags(e.X, true, values, next) || satisfyVariantTags(e.Y, true, values, next)
	default:
		panic(fmt.Sprintf("unexpected build constraint %T", expr))
	}
}
