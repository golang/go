// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !wasm || !wasm.stackswitch

package runtime

// wasmContPerG is the per-g state of WebAssembly stack switching
// (GOWASM=stackswitch). It's empty in every other
// build, so it adds nothing to the g.
type wasmContPerG struct{}

func wasmContAlloc(gp *g) {}
