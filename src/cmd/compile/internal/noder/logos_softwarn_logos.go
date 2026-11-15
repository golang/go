// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/logos/lint"
	"cmd/compile/internal/syntax"
)

var logosSoft []lint.Reported

func logosDeferSoftWarn(err error) bool {
	r, ok := lint.ReportedOf(err)
	if ok {
		logosSoft = append(logosSoft, r)
	}
	return ok
}

func logosTakeSoft() []lint.Reported {
	r := logosSoft
	logosSoft = nil
	return r
}

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
