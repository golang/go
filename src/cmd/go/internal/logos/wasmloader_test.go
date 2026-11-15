// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"strings"
	"testing"

	"cmd/go/internal/cfg"
)

const wasmLoaderBudget = 1400

func BenchmarkWasmLoaderSize(b *testing.B) {
	var n int
	for b.Loop() {
		n = len(cfg.WasmLoader)
	}
	b.ReportMetric(float64(n), "bytes")
}

func TestWasmLoaderWithinBudget(t *testing.T) {
	if n := len(cfg.WasmLoader); n == 0 || n > wasmLoaderBudget {
		t.Fatalf("loader is %d bytes, want 1 to %d", n, wasmLoaderBudget)
	}
}

func TestWasmLoaderRegistered(t *testing.T) {
	if cfg.WasmLoader != wasmLoaderText {
		t.Fatal("cfg.WasmLoader is not the loader text")
	}
}

func TestWasmLoaderContract(t *testing.T) {
	for _, part := range []string{
		`import "./wasm_exec.js";`,
		"export async function instantiateLogos(wasmBytes, imports)",
		"const getMem = () => new Uint8Array((wasmExports.mem ?? wasmExports.memory).buffer);",
		"throw new Error(readErrorString(wasmExports, getMem))",
		"wasmExports._logos_errmsg_ptr()",
		"wasmExports._logos_errmsg_len()",
		"wasmExports._logos_scratch_alloc(",
		"wasmExports._logos_scratch_reset(",
	} {
		if !strings.Contains(cfg.WasmLoader, part) {
			t.Errorf("loader lacks %q", part)
		}
	}
	if n := strings.Count(cfg.WasmLoader, "wasmExports.mem"); n != 2 {
		t.Errorf("loader names the memory export in %d places, want only the getMem closure", n)
	}
	if n := strings.Count(cfg.WasmLoader, "getMem()"); n < 3 {
		t.Errorf("loader calls getMem() in %d places, want a fresh view for every memory access", n)
	}
}
