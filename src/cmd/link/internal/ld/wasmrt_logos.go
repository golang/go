// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import "cmd/internal/sys"

const logosRTPkg = "logos/rt"

func loadLogosWasmRT(ctxt *Link) {
	if ctxt.Arch.Family != sys.Wasm || ctxt.PackageFile[logosRTPkg] == "" {
		return
	}
	loadinternal(ctxt, logosRTPkg)
}
