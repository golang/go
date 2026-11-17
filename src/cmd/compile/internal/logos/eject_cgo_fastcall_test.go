// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"go/format"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"cmd/compile/internal/syntax"
)

func TestEjectCgoFastcall(t *testing.T) {
	for _, test := range []struct {
		name    string
		imports string
		mapped  bool
	}{
		{"block", "/*\n//logos:cgo_fastcall leaf\n*/\nimport \"C\"\n", true},
		{"line", "// //logos:cgo_fastcall leaf\nimport \"C\"\n", true},
		{"group", "import (\n/*//logos:cgo_fastcall leaf*/\n\"C\"\n)\n", true},
		{"split_block", "/*//logos:cgo_fast*/ /*call leaf*/\nimport \"C\"\n", true},
		{"split_line", "/*//logos:cgo_fast*/ //call leaf\nimport \"C\"\n", true},
		{"carriage_return", "/*\r\n//logos:cgo_fastcall lea\rf\r\n*/\r\nimport \"C\"\r\n", true},
		{"ordinary_comment", "//logos:cgo_fastcall leaf\nimport \"C\"\n", false},
		{"detached", "/*\n//logos:cgo_fastcall leaf\n*/\n\nimport \"C\"\n", false},
		{"malformed", "/*\n//logos:cgo_fastcall leaf extra\n*/\nimport \"C\"\n", false},
		{"joined_suffix", "/*//logos:cgo_fastcall leaf*/ /* extra */\nimport \"C\"\n", false},
	} {
		t.Run(test.name, func(t *testing.T) {
			src := []byte("package p\n\n" + test.imports + "\nfunc f() {}\n")
			base := syntax.NewFileBase("bridge.lgo")
			file := ejectParse(t, base, src, 0)
			out := t.TempDir()
			if err := Eject(out, []*syntax.File{file}, map[*syntax.File][]byte{file: src}, nil); err != nil {
				t.Fatal(err)
			}
			got, err := os.ReadFile(filepath.Join(out, "bridge.go"))
			if err != nil {
				t.Fatal(err)
			}
			for _, formatted := range []bool{false, true} {
				if formatted {
					got, err = format.Source(got)
					if err != nil {
						t.Fatal(err)
					}
				}
				preamble := cgoPreamble(t, got)
				for _, directive := range []string{"#cgo noescape leaf", "#cgo nocallback leaf"} {
					if strings.Contains(preamble, directive) != test.mapped {
						t.Errorf("formatted=%v: mapping %q = %v, want %v\n%s", formatted, directive, !test.mapped, test.mapped, got)
					}
				}
			}
		})
	}
}
