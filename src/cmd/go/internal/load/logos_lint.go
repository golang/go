// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import (
	"cmd/go/internal/cfg"
	"cmd/go/internal/logos/config"
	"cmd/go/internal/modload"
)

// logosLintMark is the setToolFlags hook for the lints (M9). In logos mode it
// records on every package of a main module the -logos.cfg policy its
// logos.toml [lints] table resolves to, with the --pedantic flag applied.
// Standard-library packages, the generated test main, packages of other
// modules (third-party code gets no code lints), and packages whose
// logos.toml does not parse keep a nil policy and are never linted. The
// build decides per action whether the policy reaches the compiler: a
// loGo/s package passes it on every compile, a Go package only through the
// lint action of logos lint.
func logosLintMark(ld *modload.Loader, p *Package) {
	p.Internal.LogosLint = nil
	if !cfg.Logos || p.Standard || p.Internal.TestmainGo != nil {
		return
	}
	if p.Module == nil || !p.Module.Main {
		return
	}
	cf, err := logosConfigAt(logosConfigRootForPackage(ld, p))
	if err != nil || cf == nil {
		return
	}
	policy, err := cf.Lints.PolicyJSON(logosPedanticMode())
	if err != nil {
		return
	}
	p.Internal.LogosLint = policy
}

// logosPedanticMode returns the state of the --pedantic flag.
func logosPedanticMode() config.PedanticMode {
	switch {
	case !cfg.PedanticSet:
		return config.PedanticDefault
	case cfg.Pedantic:
		return config.PedanticOn
	}
	return config.PedanticOff
}
