// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"internal/buildcfg"
	"internal/logosfile"
	"strings"
)

func logosFastCall(p *Package, n *Name) bool {
	if *gccgo {
		return false
	}
	return p.noEscapes[n.C] && p.noCallbacks[n.C]
}

func logosStableFastCall(p *Package, n *Name) bool {
	return buildcfg.Experiment.LogosFastCgo && logosFastCall(p, n)
}

func logosCgoCall(p *Package, n *Name) string {
	if logosStableFastCall(p, n) {
		return "_cgo_runtime_cgocall_fast"
	}
	return "_cgo_runtime_cgocall"
}

const logosFastcallMarker = "//logos:cgo_fastcall"

func logosMapFastcallDirectives(f *File, srcfile string) {
	if !logosfile.Is(srcfile) {
		return
	}
	if f.NoCallbacks == nil {
		f.NoCallbacks = make(map[string]bool)
	}
	if f.NoEscapes == nil {
		f.NoEscapes = make(map[string]bool)
	}
	lines := strings.Split(f.Preamble, "\n")
	for i, line := range lines {
		fields := strings.Fields(line)
		if len(fields) != 2 || fields[0] != logosFastcallMarker {
			continue
		}
		f.NoCallbacks[fields[1]] = true
		f.NoEscapes[fields[1]] = true
		lines[i] = ""
	}
	f.Preamble = strings.Join(lines, "\n")
}
