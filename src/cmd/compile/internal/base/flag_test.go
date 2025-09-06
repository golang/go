// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package base

import (
	"cmd/internal/obj"
	"strings"
	"testing"
)

func BenchmarkParseSpectreNew(b *testing.B) {
	if Ctxt == nil {
		Ctxt = &obj.Link{}
	}

	testCases := []struct {
		name  string
		input string
	}{{
		name:  "empty",
		input: "",
	}, {
		name:  "index",
		input: "index",
	}, {
		name:  "ret",
		input: "ret",
	}, {
		name:  "index_ret",
		input: "index,ret",
	}, {
		name:  "all",
		input: "all",
	}, {
		name:  "multiple_indices_ret",
		input: strings.Repeat("index,", 10) + "ret",
	}}

	for _, tc := range testCases {
		b.Run(tc.name, func(b *testing.B) {
			// Reset variables before each run
			oldFlagCfgSpectreIndex := Flag.Cfg.SpectreIndex
			oldCtxtRetpoline := Ctxt.Retpoline
			defer func() {
				Flag.Cfg.SpectreIndex = oldFlagCfgSpectreIndex
				Ctxt.Retpoline = oldCtxtRetpoline
			}()

			b.ResetTimer()
			for b.Loop() {
				parseSpectre(tc.input)
			}
		})
	}
}
