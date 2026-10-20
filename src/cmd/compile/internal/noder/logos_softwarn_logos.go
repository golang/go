// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/logos/lint"
	"cmd/compile/internal/syntax"
)

// M6-S3 soft diagnostics (SPEC Sec 1.2): warn-severity loGo/s rules do not
// fail the check. The predicate matches types2's firstErr exemption exactly,
// so upstream soft errors (duplicate labels, invalid type arguments) keep
// failing through ErrorfAt. The warnings wait here until the lint pass
// resolves them (M9), so the severity of logos.toml and //logos:allow apply
// to them like to every other lint.

// logosSoft holds the warnings of types2 that no lint pass has resolved yet.
var logosSoft []lint.Reported

// logosDeferSoftWarn records a warn-severity loGo/s diagnostic of types2 for
// the lint pass and reports true. Anything else returns false for the
// upstream path.
func logosDeferSoftWarn(err error) bool {
	r, ok := lint.ReportedOf(err)
	if ok {
		logosSoft = append(logosSoft, r)
	}
	return ok
}

// logosTakeSoft returns the recorded warnings and forgets them.
func logosTakeSoft() []lint.Reported {
	r := logosSoft
	logosSoft = nil
	return r
}

// logosFlushSoftWarns reports the recorded warnings of a check that no lint
// pass follows: a package that failed to check, or a compile with no policy
// or no lints. They resolve under the -logos.cfg policy, or the default
// policy when the compile has none.
func logosFlushSoftWarns(m *posMap, files []*syntax.File, checkErr error) {
	if len(logosSoft) == 0 || (checkErr == nil && logosLintActive()) {
		return
	}
	policy := lint.DefaultPolicy()
	if base.LogosCfg != "" {
		policy = logosLintPolicy()
	}
	lintable, sources := logosLintInputs(files)
	in := &lint.Input{
		Files:    lintable,
		Sources:  sources,
		Kind:     lint.KindLgo,
		Policy:   policy,
		Reported: logosTakeSoft(),
	}
	for _, r := range lint.Resolve(in) {
		base.LogosReport(logosDiag(m, r))
	}
}
