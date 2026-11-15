// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import (
	"cmd/go/internal/cfg"
	"cmd/go/internal/logos/settings"
	"cmd/go/internal/modload"
)

func logosLintMark(ld *modload.Loader, p *Package) {
	p.Internal.LogosLint = nil
	if !cfg.Logos || p.Standard || p.Internal.TestmainGo != nil {
		return
	}
	if p.Module == nil || !p.Module.Main {
		return
	}
	p.Internal.LogosLint = settings.Current().LintPolicy(settings.ModuleAt(logosConfigRootForPackage(ld, p)))
}

func logosLintXTest(p, pxtest *Package) {
	if pxtest != nil {
		pxtest.Internal.LogosLint = p.Internal.LogosLint
	}
}
