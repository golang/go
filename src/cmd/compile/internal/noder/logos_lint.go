// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"os"

	"cmd/compile/internal/base"
	"cmd/compile/internal/logos/lint"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"cmd/internal/src"
)

// logosLintHook runs the lints of the checked package (SPEC Sec 1.2, M9). It
// runs once after the first successful check, before lowering, so the rules
// see loGo/s syntax and types. A loGo/s package lints on every compile that
// names a policy with -logos.cfg; a Go package lints only under
// -logos.lintonly, which only logos lint passes. Findings that fail the
// compile stop it here, except under -logos.lintonly, where the compile
// continues to escape analysis so every finding of the package prints
// together.
func logosLintHook(m *posMap, files []*syntax.File, pkg *types2.Package, info *types2.Info) {
	if base.LogosCfg == "" || (!base.Logos && !base.LogosLintOnly) {
		return
	}
	policy := logosLintPolicy()
	kind := lint.KindGo
	if base.Logos {
		kind = lint.KindLgo
	}
	lintable, sources := logosLintInputs(files)
	results := lint.Run(&lint.Input{
		Pkg:     pkg,
		Info:    info,
		Files:   lintable,
		Sources: sources,
		Kind:    kind,
		Policy:  policy,
	})
	for _, r := range results {
		base.LogosReport(logosDiag(m, r))
	}
	if !base.LogosLintOnly {
		base.ExitIfErrors()
	}
}

// logosLintPolicy reads the -logos.cfg policy. A policy the go command
// wrote and this compiler cannot read is a build error.
func logosLintPolicy() *lint.Policy {
	data, err := os.ReadFile(base.LogosCfg)
	if err != nil {
		base.ErrorfAt(src.NoXPos, 0, "logos: %v", err)
		base.ErrorExit()
	}
	policy, err := lint.ParsePolicy(data)
	if err != nil {
		base.ErrorfAt(src.NoXPos, 0, "logos: -logos.cfg %s: %v", base.LogosCfg, err)
		base.ErrorExit()
	}
	return policy
}

// logosLintInputs returns the files to lint, without compiler-generated cgo
// outputs, and the original bytes the parser stashed for them.
func logosLintInputs(files []*syntax.File) ([]*syntax.File, map[*syntax.File][]byte) {
	stashed, _ := nextInputs(files)
	var lintable []*syntax.File
	for _, f := range files {
		if !isCgoOutput(f.Pos().RelFilename()) {
			lintable = append(lintable, f)
		}
	}
	return lintable, stashed
}

// logosDiag converts a lint result to the compiler diagnostic sink.
func logosDiag(m *posMap, r lint.Result) base.LogosDiag {
	d := base.LogosDiag{
		Rule:         r.Rule,
		Severity:     r.Severity.String(),
		Pos:          m.makeXPos(r.Pos),
		End:          m.makeXPos(r.End),
		Message:      r.Message,
		OriginExpr:   r.Provenance.OriginExpr,
		OriginLine:   r.Provenance.OriginLine,
		InferredType: r.Provenance.InferredType,
	}
	for _, f := range r.Fixes {
		d.Fixes = append(d.Fixes, base.LogosFix(f))
	}
	return d
}
