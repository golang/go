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
	for name := range repos {
		b.Run(name, func(b *testing.B) {
			env := getRepo(b, name).sharedEnv(b)
			symbols := env.Symbol(*symbolQuery) // warm the cache

			if testing.Verbose() {
				fmt.Println("Results:")
				for i, symbol := range symbols {
					fmt.Printf("\t%d. %s (%s)\n", i, symbol.Name, symbol.ContainerName)
				}
			}

			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				env.Symbol(*symbolQuery)
			}
		})
	}
}
