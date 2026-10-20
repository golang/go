// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
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

func logosHashLinkPolicy(h io.Writer) {
	capabilities.HashLinkPolicy(h)
}

func logosLinkCapFlag(ldflags []string, objdir string) []string {
	return capabilities.AppendLinkCapFlag(ldflags, objdir)
}
