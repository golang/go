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
	"cmd/go/internal/logos/settings"
)

func addWasmLoaderFlag(fs *flag.FlagSet) {
	fs.BoolVar(&settings.CommandLine.WasmLoader, "wasm-loader", true, "")
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
