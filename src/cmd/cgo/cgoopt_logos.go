// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"internal/buildcfg"
	"maps"
)

// logosStableCgoFrame reports whether the result frame stays at its original
// address until the C wrapper returns. Without a callback, cgocall remains in
// syscall state throughout that interval, preventing stack copying or shrinking.
// This does not prove noescape or change the runtime entry and callback guards.
func logosStableCgoFrame(p *Package, n *Name) bool {
	if *gccgo {
		return false
	}
	return logosStableFastCall(p, n) ||
		buildcfg.Experiment.LogosCgoOpt && p.noCallbacks[n.C]
}

// logosCollectCgoContracts makes the already parsed package contracts available
// to C wrappers emitted before the file that supplied a directive. Record still
// owns names, types, declarations, and diagnostics in their original order.
func logosCollectCgoContracts(p *Package, files []*File) {
	if !buildcfg.Experiment.LogosCgoOpt || *gccgo || *godefs {
		return
	}
	for _, f := range files {
		maps.Copy(p.noCallbacks, f.NoCallbacks)
		maps.Copy(p.noEscapes, f.NoEscapes)
	}
}
