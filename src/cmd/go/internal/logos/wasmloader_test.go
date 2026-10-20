// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"os"
	"strings"
	"testing"

	"cmd/go/internal/cfg"
)

// wasmLoaderBudget is the loader size target of SPEC Sec 14.2 (1.4 KB,
// excluding wasm_exec.js), counted in bytes.
const wasmLoaderBudget = 1400

// BenchmarkWasmLoaderSize measures the loader bytes that a wasm build
// writes beside its output, the text in cfg.WasmLoader. It fails when the
// loader exceeds the 1.4 KB budget and reports the size as a metric.
func BenchmarkWasmLoaderSize(b *testing.B) {
	var n int
	for b.Loop() {
		n = len(cfg.WasmLoader)
	}
	b.ReportMetric(float64(n), "bytes")
	if n == 0 {
		b.Fatal("cfg.WasmLoader is empty: the loader is not registered")
	}
	if n > wasmLoaderBudget {
		b.Fatalf("loader is %d bytes, want at most %d", n, wasmLoaderBudget)
	}
}

// TestWasmLoaderWithinBudget fails a plain go test run, not only a
// benchmark run, when the loader outgrows the 1.4 KB budget.
func TestWasmLoaderWithinBudget(t *testing.T) {
	if n := len(cfg.WasmLoader); n == 0 || n > wasmLoaderBudget {
		t.Fatalf("loader is %d bytes, want 1 to %d", n, wasmLoaderBudget)
	}
}

// TestWasmLoaderTextMatchesTemplate pins the Go constant to the template
// file: they are one loader in two spellings.
func TestWasmLoaderTextMatchesTemplate(t *testing.T) {
	want, err := os.ReadFile("wasmloader.mjs.tmpl")
	if err != nil {
		t.Fatal(err)
	}
	if string(want) != wasmLoaderText {
		t.Fatal("wasmLoaderText differs from wasmloader.mjs.tmpl; mirror the template into wasmloader.go")
	}
}

// TestWasmLoaderRegistered pins the registration that lets cmd/go/internal/work
// write the loader without importing this package.
func TestWasmLoaderRegistered(t *testing.T) {
	if cfg.WasmLoader != wasmLoaderText {
		t.Fatal("cfg.WasmLoader is not the loader text")
	}
}

// TestWasmLoaderContract pins the pieces of the loader that SPEC Sec 14.2
// names: the wasm_exec.js startup, the instantiateLogos export, the
// memory.buffer re-read through getMem, the errCode check, and the error
// text read from wasm memory.
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
