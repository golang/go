// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"flag"
	"fmt"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/load"
)

// addWasmLoaderFlag registers --wasm-loader (SPEC Sec 14.2, R-U1) on one
// build flag set. It has no help text, so "go help build" keeps the
// upstream bytes (Sec 0.1). Its default is true under logos and false
// under plain go until the run.bash flip.
func addWasmLoaderFlag(fs *flag.FlagSet) {
	fs.BoolVar(&cfg.BuildWasmLoader, "wasm-loader", cfg.Logos, "")
}

// writeWasmLoader writes the host loader beside the .wasm output of the
// finished build of p when the build emits one, and does nothing when the
// build failed or p emits no loader. It goes through the shell, so -n and
// -x show the write and -n skips it.
func (b *Builder) writeWasmLoader(p *load.Package) {
	path, ok := load.WasmLoaderFor(p)
	if !ok || base.GetExitStatus() != 0 {
		return
	}
	if err := b.BackgroundShell().writeFile(path, []byte(cfg.WasmLoader)); err != nil {
		base.Fatal(fmt.Errorf("writing wasm loader: %w", err))
	}
}
