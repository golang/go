// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bench

import (
	"flag"
	"testing"

	. "golang.org/x/tools/internal/lsp/regtest"
)

var iwlOptions struct {
	workdir string
}

func init() {
	flag.StringVar(&iwlOptions.workdir, "iwl_workdir", "", "if set, run IWL benchmark in this directory")
}

func TestBenchmarkIWL(t *testing.T) {
	if iwlOptions.workdir == "" {
		t.Skip("-iwl_workdir not configured")
	}

	opts := stressTestOptions(iwlOptions.workdir)
	// Don't skip hooks, so that we can wait for IWL.
	opts = append(opts, SkipHooks(false))

	results := testing.Benchmark(func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			WithOptions(opts...).Run(t, "", func(t *testing.T, env *Env) {})

		}
	})

	printBenchmarkResults(results)
}
