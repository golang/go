// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"fmt"
	"io"

	"cmd/go/internal/cfg"
	"cmd/go/internal/load"
	"cmd/go/internal/logos/capabilities"
	"cmd/go/internal/modload"
)

func logosSnapshotLinkPolicy(ld *modload.Loader) {
	if !cfg.Logos {
		return
	}
	cf, mains, err := load.MainModuleLinkPolicy(ld)
	capabilities.SnapshotLinkPolicy(cf, mains, err)
}

func logosHashLinkPolicy(h io.Writer, p *load.Package) {
	capabilities.HashLinkPolicy(h)
	if origin := logosTestPackageOrigin(p); origin != "" {
		fmt.Fprintf(h, "logostestpackage %q\n", origin)
	}
}

func logosLinkCapFlag(ldflags []string, objdir string, p *load.Package) []string {
	ldflags = capabilities.AppendLinkCapFlag(ldflags, objdir)
	if origin := logosTestPackageOrigin(p); origin != "" {
		ldflags = append(ldflags, "-logos.test-package="+origin)
	}
	return ldflags
}

// logosTestPackageOrigin identifies the synthetic external test import from
// loader metadata, so an ordinary package ending in _test keeps its identity.
func logosTestPackageOrigin(p *load.Package) string {
	if !cfg.Logos || p == nil || p.Internal.TestmainGo == nil {
		return ""
	}
	for _, q := range p.Internal.Imports {
		if q.ForTest != "" && p.ImportPath == q.ForTest+".test" && q.ImportPath == q.ForTest+"_test" {
			return q.ForTest
		}
	}
	return ""
}
