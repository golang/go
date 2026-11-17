// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"strings"
	"testing"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// Anchor: Tony Hoare. Later local type specifications must not change the
// bindings visible when a field default is declared. All affected types stay
// unconstructed: this tests declaration checking, not default application.
func TestStructDefaultGroupScope(t *testing.T) {
	for _, mode := range []struct {
		name   string
		types  bool
		syntax bool
	}{
		{name: "without Types"},
		{name: "with Types", types: true},
		{name: "StoreTypesInSyntax", syntax: true},
	} {
		t.Run(mode.name, func(t *testing.T) {
			for _, tc := range []struct {
				name string
				src  string
				want string
			}{
				{
					name: "later local type",
					src: `package p
const C = 17
func f() {
	type (
		Cfg struct { N int = C }
		C int
	)
}
`,
				},
				{
					name: "later local alias",
					src: `package p
func f() {
	const C = 17
	{
		type (
			Cfg struct { N int = C }
			C = int
		)
	}
}
`,
				},
				{
					name: "later type shadows import alias",
					src: `package p
import clock "time"
func f() {
	type (
		Cfg struct { Delay clock.Duration = clock.Second }
		clock int
	)
}
`,
				},
				{
					name: "later type shadows map key",
					src: `package p
const Key = 17
func f() {
	type (
		Cfg struct { Values map[int]int = map[int]int{Key: 1} }
		Key int
	)
}
`,
				},
				{
					name: "package forward constant",
					src: `package p
type Cfg struct { N int = C }
const C = 17
`,
				},
				{
					name: "earlier local type conversion",
					src: `package p
const C = 17
func f() {
	type (
		Value int
		Cfg struct { N Value = Value(C) }
	)
}
`,
				},
				{
					name: "completed recursive generic type and alias",
					src: `package p
func f() {
	type (
		Node[T ~int] struct {
			Next *Node[T]
			N T = 17
		}
		Alias = Node[int]
	)
}
`,
				},
				{
					name: "earlier local type is not a constant",
					src: `package p
const C = 17
func f() {
	type (
		C int
		Cfg struct { N int = C }
	)
}
`,
					want: "is not an expression",
				},
				{
					name: "future local type is undefined",
					src: `package p
func f() {
	type (
		Cfg struct { N int = Missing }
		Missing int
	)
}
`,
					want: "undefined: Missing",
				},
			} {
				t.Run(tc.name, func(t *testing.T) {
					file := ejectParse(t, syntax.NewFileBase("defaultgroupscope.lgo"), []byte(tc.src), syntax.LogosMode)
					info := &types2.Info{StoreTypesInSyntax: mode.syntax}
					if mode.types {
						info.Types = make(map[syntax.Expr]types2.TypeAndValue)
					}
					var errors []string
					conf := types2.Config{
						Logos:    true,
						Importer: &ejectImporter{pkgs: make(map[string]*types2.Package)},
						Error:    func(err error) { errors = append(errors, err.Error()) },
					}
					_, err := conf.Check("p", []*syntax.File{file}, info)
					if tc.want == "" {
						if err != nil {
							t.Fatalf("valid declaration-scope default rejected: %v", errors)
						}
						return
					}
					if err == nil || !strings.Contains(strings.Join(errors, "\n"), tc.want) {
						t.Fatalf("field default errors = %v, want %q without constructing Cfg", errors, tc.want)
					}
				})
			}
		})
	}
}
