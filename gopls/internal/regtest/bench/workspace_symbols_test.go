// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bench

import (
	"flag"
	"fmt"
	"testing"
)

var symbolQuery = flag.String("symbol_query", "test", "symbol query to use in benchmark")

// BenchmarkWorkspaceSymbols benchmarks the time to execute a workspace symbols
// request (controlled by the -symbol_query flag).
func BenchmarkWorkspaceSymbols(b *testing.B) {
	env := benchmarkEnv(b)

	// Make an initial symbol query to warm the cache.
	symbols := env.WorkspaceSymbol(*symbolQuery)

	if testing.Verbose() {
		fmt.Println("Results:")
		for i := 0; i < len(symbols); i++ {
			fmt.Printf("\t%d. %s (%s)\n", i, symbols[i].Name, symbols[i].ContainerName)
		}
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		env.WorkspaceSymbol(*symbolQuery)
	}
}
