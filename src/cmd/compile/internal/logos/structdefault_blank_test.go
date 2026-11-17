// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"strings"
	"testing"
)

// Blank fields may repeat with different types. Each default must be checked
// against its own declaration, including a field group beginning with _.
func TestStructDefaultBlankFieldIdentity(t *testing.T) {
	for _, tc := range []struct {
		name string
		decl string
		want string
	}{
		{"valid second blank", `_ int; _ string = "ok"`, ""},
		{"invalid second blank", `_ string; _ int = "bad"`, "cannot use"},
		{"valid blank defaults", `_ int = 1; _ string = "ok"`, ""},
		{"invalid blank defaults", `_ string = "first"; _ int = "bad"`, "cannot use"},
		{"valid group beginning with blank", `_ int; _, Value string = "ok"`, ""},
		{"invalid group beginning with blank", `_ string; _, Value int = "bad"`, "cannot use"},
		{"valid group ending with blank", `_ int; Value, _ string = "ok"`, ""},
		{"valid distinct names", `First int = 1; Second string = "ok"`, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			src := "package p\ntype Unused struct { " + tc.decl + " }\n"
			file := ejectParse(t, syntax.NewFileBase("defaultblank.lgo"), []byte(src), syntax.LogosMode)
			var errors []string
			conf := types2.Config{
				Logos: true,
				Error: func(err error) { errors = append(errors, err.Error()) },
			}
			_, err := conf.Check("p", []*syntax.File{file}, &types2.Info{})
			if tc.want == "" {
				if err != nil {
					t.Fatalf("valid field default rejected: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(strings.Join(errors, "\n"), tc.want) {
				t.Fatalf("field default errors = %v, want %q without constructing Unused", errors, tc.want)
			}
		})
	}
}
