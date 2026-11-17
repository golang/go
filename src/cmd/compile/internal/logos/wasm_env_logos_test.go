// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"internal/testenv"
	"strings"
	"testing"
)

func TestWasmNodeIsolatesParentEnvironment(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	wasmNode(t)
	// The host may have a large CI environment. It is not input to the
	// fixture and must neither overflow wasm's startup area nor leak in.
	t.Setenv("LOGOS_WASM_PARENT_ENV", strings.Repeat("x", 16<<10))
	const source = `package main

import (
	"fmt"
	"os"
	"runtime"
)

func main() {
	_, inherited := os.LookupEnv("LOGOS_WASM_PARENT_ENV")
	fmt.Println(inherited, runtime.GOMAXPROCS(0))
}
`
	wasm := buildWasm(t, testenv.GoToolPath(t), "js", source, "")
	got, err := runWasmNode(t, wasm)
	if err != nil || got != "false 1\n" {
		t.Fatalf("isolated wasm environment: run %v, output %q; want %q", err, got, "false 1\n")
	}
}
