// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm64asm

import (
	"encoding/hex"
	"io/ioutil"
	"strings"
	"testing"
)

func TestDecode(t *testing.T) {
	data, err := ioutil.ReadFile("testdata/cases.txt")
	if err != nil {
		t.Fatal(err)
	}
	all := string(data)
	for strings.Contains(all, "\t\t") {
		all = strings.Replace(all, "\t\t", "\t", -1)
	}
	for _, line := range strings.Split(all, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		f := strings.SplitN(line, "\t", 3)
		i := strings.Index(f[0], "|")
		if i < 0 {
			t.Errorf("parsing %q: missing | separator", f[0])
			continue
		}
		if i%2 != 0 {
			t.Errorf("parsing %q: misaligned | separator", f[0])
		}
		code, err := hex.DecodeString(f[0][:i] + f[0][i+1:])
		if err != nil {
			t.Errorf("parsing %q: %v", f[0], err)
			continue
		}
		syntax, asm := f[1], f[2]
		inst, decodeErr := Decode(code)
		if decodeErr != nil && decodeErr != errUnknown {
			// Some rarely used system instructions are not supported
			// Following logicals will filter such unknown instructions

			t.Errorf("parsing %x: %s", code, decodeErr)
			continue
		}
		var out string
		switch syntax {
		case "gnu":
			out = GNUSyntax(inst)
		case "plan9":
			out = GoSyntax(inst, 0, nil, nil)
		default:
			t.Errorf("unknown syntax %q", syntax)
			continue
		}
		// TODO: system instruction.
		var Todo = strings.Fields(`
			sys
			dc
			at
			tlbi
			ic
			hvc
			smc
		`)
		if strings.Replace(out, " ", "", -1) != strings.Replace(asm, " ", "", -1) && !hasPrefix(asm, Todo...) {
			// Exclude MSR since GNU objdump result is incorrect. eg. 0xd504431f msr s0_4_c4_c3_0, xzr
			if !strings.HasSuffix(asm, " nv") && !strings.HasPrefix(asm, "msr") {
				t.Errorf("Decode(%s) [%s] = %s, want %s", f[0], syntax, out, asm)
			}
		}
	}
}
