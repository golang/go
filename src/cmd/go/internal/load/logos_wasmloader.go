// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import (
	"strings"

	"cmd/go/internal/cfg"
)

const (
	wasmOutputSuffix = ".wasm"
	wasmLoaderSuffix = ".loader.mjs"
)

// wantsWasmLoader reports the build-wide part of the loader decision (SPEC
// Sec 14.2, R-U1): the --wasm-loader flag is on, the build command is
// build, the target is GOARCH=wasm, and p is a main package. It reads the
// build configuration, never the module kind, so a Go main package and a
// loGo/s main package behave alike.
func wantsWasmLoader(p *Package) bool {
	return cfg.BuildWasmLoader &&
		cfg.CmdName == "build" &&
		cfg.BuildContext.GOARCH == "wasm" &&
		p.Name == "main"
}

// isWasmFile reports whether out names a .wasm file.
func isWasmFile(out string) bool {
	return strings.HasSuffix(out, wasmOutputSuffix)
}

// EmitsWasmLoader reports at load time whether building p writes the host
// loader beside its output: the build-wide conditions hold and -o names a
// .wasm file. The load of p adds the implicit logos/rt import when it
// holds, so the exported helpers the loader calls are linked in. The
// default output name of a wasm main package carries no .wasm suffix, so
// a build without -o writes no loader.
func EmitsWasmLoader(p *Package) bool {
	return wantsWasmLoader(p) && isWasmFile(cfg.BuildO)
}

// WasmLoaderFor returns the loader file that goes beside the built output
// of p, and reports whether the build writes one: the build-wide
// conditions hold and the target of p is a .wasm file. The loader name is
// the target with its .wasm suffix replaced by .loader.mjs.
func WasmLoaderFor(p *Package) (string, bool) {
	if !wantsWasmLoader(p) || !isWasmFile(p.Target) {
		return "", false
	}
	return strings.TrimSuffix(p.Target, wasmOutputSuffix) + wasmLoaderSuffix, true
}
