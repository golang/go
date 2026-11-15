// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"internal/buildcfg"

	"cmd/asm/internal/flags"
	"cmd/internal/obj"
)

var wasmStackSwitch = flag.Bool("logos.wasmstackswitch", false, "assemble for wasm stack switching")

func logosWasmStackSwitch(ctxt *obj.Link) {
	if !*wasmStackSwitch || buildcfg.GOARCH != "wasm" {
		return
	}
	ctxt.WasmStackSwitch = true
	flags.D = append(flags.D, "GOWASM_stackswitch")
}
