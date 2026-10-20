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

// This file holds the loGo/s facts-sidecar writer (M3-S1). Upstream files
// carry only hook lines; all logic lives here.
//
// writeLogosFacts appends the __.LOGOS member right after __.PKGDEF in
// compiler-object mode. The member binds to the __.PKGDEF linker
// fingerprint that WriteExports just stored in base.Ctxt.Fingerprint, so an
// importer rejects stale facts cleanly. The linker already skips non-.o
// members (cmd/link/internal/ld/lib.go), so no linker change is needed.
// Until S2-S7 populate records through facts.SetCurrent, the member
// carries the header and fingerprint only. The member is emitted only
// for loGo/s packages and, for the MayYield facts of the WASM profile
// (M16-S1, T-Sec6), for every package built for GOARCH=wasm: a plain-Go
// archive on any other target stays byte-identical to upstream, and
// every member-counting reader (pack's object check, nm, objdump) sees
// the same stream it always has.
func writeLogosFacts(bout *bio.Writer) {
	if !base.Logos && !wasmyield.Active() {
		return
	}
	start := startArchiveEntry(bout)
	bout.Write(facts.Encode(base.Ctxt.Fingerprint, facts.Current()))
	finishArchiveEntry(bout, start, facts.MemberName)
}

// logosWasmFinish is the M16-S1 hook after the backend loop: it solves the
// MayYield facts of the package and assembles the functions held for them.
func logosWasmFinish() {
	wasmyield.Finish()
}
