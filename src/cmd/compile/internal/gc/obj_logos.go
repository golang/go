// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/logos/facts"
	"cmd/compile/internal/logos/wasmyield"
	"cmd/internal/bio"
)

func anyFnFact(pred func(*facts.FnFact) bool) bool {
	f := facts.Current()
	if f == nil {
		return false
	}
	for i := range f.Fns {
		if pred(&f.Fns[i]) {
			return true
		}
	}
	return false
}

func hasLogosAssertFacts() bool {
	return anyFnFact(func(fn *facts.FnFact) bool { return fn.HasAssert })
}

func hasLogosLintFacts() bool {
	if f := facts.Current(); f != nil && (len(f.AllocFrees) != 0 || len(f.Locks) != 0) {
		return true
	}
	return anyFnFact(func(fn *facts.FnFact) bool { return fn.AlwaysNilError })
}

func hasLogosConstFacts() bool {
	f := facts.Current()
	if f == nil {
		return false
	}
	for i := range f.Fns {
		if f.Fns[i].ParamMutated != "" {
			return true
		}
	}
	return false
}

func writeLogosFacts(bout *bio.Writer) {
	if !base.Logos && !wasmyield.Active() && !hasLogosAssertFacts() && !hasLogosLintFacts() && !hasLogosConstFacts() {
		return
	}
	start := startArchiveEntry(bout)
	bout.Write(facts.Encode(base.Ctxt.Fingerprint, facts.Current()))
	finishArchiveEntry(bout, start, facts.MemberName)
}
