// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build js || wasip1

package runtime

// logosWasmMaxMemory is the 4 GiB limit of a wasm32 linear memory, the most
// memory a wasm program can reach: wasm has no physical memory to read.
const logosWasmMaxMemory = 1 << 32

// logosPhysicalMemory returns the wasm32 memory limit.
func logosPhysicalMemory() uint64 { return logosWasmMaxMemory }
