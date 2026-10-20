// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import "cmd/internal/sys"

const logosRTPkg = "logos/rt"

// loadLogosWasmRT loads logos/rt on GOARCH=wasm when the import
// configuration lists it (SPEC Sec 14.2). The go command lists logos/rt for
// a wasm build that emits the host loader, so the //go:wasmexport helpers
// that the loader calls must link even when no object of the program
// references the package: a main package that holds only the implicit
// import records no reference to it, and the linker loads a library only
// through a reference. A target other than wasm, and an import
// configuration without logos/rt, load nothing.
func loadLogosWasmRT(ctxt *Link) {
	if ctxt.Arch.Family != sys.Wasm || ctxt.PackageFile[logosRTPkg] == "" {
		return
	}
	loadinternal(ctxt, logosRTPkg)
}
