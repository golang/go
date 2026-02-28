// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package asm

import (
	"cmd/asm/internal/lex"
	"strings"
	"testing"
)

type badInstTest struct {
	input, error string
}

func TestAMD64BadInstParser(t *testing.T) {
	testBadInstParser(t, "amd64", []badInstTest{
		// Test AVX512 suffixes.
		{"VADDPD.A X0, X1, X2", `unknown suffix "A"`},
		{"VADDPD.A.A X0, X1, X2", `unknown suffix "A"; duplicate suffix "A"`},
		{"VADDPD.A.A.A X0, X1, X2", `unknown suffix "A"; duplicate suffix "A"`},
		{"VADDPD.A.B X0, X1, X2", `unknown suffix "A"; unknown suffix "B"`},
		{"VADDPD.Z.A X0, X1, X2", `Z suffix should be the last; unknown suffix "A"`},
		{"VADDPD.Z.Z X0, X1, X2", `Z suffix should be the last; duplicate suffix "Z"`},
		{"VADDPD.SAE.BCST X0, X1, X2", `can't combine rounding/SAE and broadcast`},
		{"VADDPD.BCST.SAE X0, X1, X2", `can't combine rounding/SAE and broadcast`},
		{"VADDPD.BCST.Z.SAE X0, X1, X2", `Z suffix should be the last; can't combine rounding/SAE and broadcast`},
		{"VADDPD.SAE.SAE X0, X1, X2", `duplicate suffix "SAE"`},
		{"VADDPD.RZ_SAE.SAE X0, X1, X2", `bad suffix combination`},

		// BSWAP on 16-bit registers is undefined. See #29167,
		{"BSWAPW DX", `unrecognized instruction`},
		{"BSWAPW R11", `unrecognized instruction`},
	})
}

func testBadInstParser(t *testing.T, goarch string, tests []badInstTest) {
	for i, test := range tests {
		arch, ctxt := setArch(goarch)
		tokenizer := lex.NewTokenizer("", strings.NewReader(test.input+"\n"), nil)
		parser := NewParser(ctxt, arch, tokenizer, false)

		err := tryParse(t, func() {
			parser.Parse()
		})

		switch {
		case err == nil:
			t.Errorf("#%d: %q: want error %q; have none", i, test.input, test.error)
		case !strings.Contains(err.Error(), test.error):
			t.Errorf("#%d: %q: want error %q; have %q", i, test.input, test.error, err)
		}
	}
}
