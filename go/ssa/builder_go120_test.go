// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.20
// +build go1.20

package ssa_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"testing"

	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"
)

func TestBuildPackageGo120(t *testing.T) {
	tests := []struct {
		name     string
		src      string
		importer types.Importer
	}{
		{"slice to array", "package p; var s []byte; var _ = ([4]byte)(s)", nil},
		{"slice to zero length array", "package p; var s []byte; var _ = ([0]byte)(s)", nil},
		{"slice to zero length array type parameter", "package p; var s []byte; func f[T ~[0]byte]() { tmp := (T)(s); var z T; _ = tmp == z}", nil},
		{"slice to non-zero length array type parameter", "package p; var s []byte; func h[T ~[1]byte | [4]byte]() { tmp := T(s); var z T; _ = tmp == z}", nil},
		{"slice to maybe-zero length array type parameter", "package p; var s []byte; func g[T ~[0]byte | [4]byte]() { tmp := T(s); var z T; _ = tmp == z}", nil},
		{
			"rune sequence to sequence cast patterns", `
			package p
			// Each of fXX functions describes a 1.20 legal cast between sequences of runes
			// as []rune, pointers to rune arrays, rune arrays, or strings.
			//
			// Comments listed given the current emitted instructions [approximately].
			// If multiple conversions are needed, these are seperated by |.
			// rune was selected as it leads to string casts (byte is similar).
			// The length 2 is not significant.
			// Multiple array lengths may occur in a cast in practice (including 0).
			func f00[S string, D string](s S)                               { _ = D(s) } // ChangeType
			func f01[S string, D []rune](s S)                               { _ = D(s) } // Convert
			func f02[S string, D []rune | string](s S)                      { _ = D(s) } // ChangeType | Convert
			func f03[S [2]rune, D [2]rune](s S)                             { _ = D(s) } // ChangeType
			func f04[S *[2]rune, D *[2]rune](s S)                           { _ = D(s) } // ChangeType
			func f05[S []rune, D string](s S)                               { _ = D(s) } // Convert
			func f06[S []rune, D [2]rune](s S)                              { _ = D(s) } // SliceToArrayPointer; Deref
			func f07[S []rune, D [2]rune | string](s S)                     { _ = D(s) } // SliceToArrayPointer; Deref | Convert
			func f08[S []rune, D *[2]rune](s S)                             { _ = D(s) } // SliceToArrayPointer
			func f09[S []rune, D *[2]rune | string](s S)                    { _ = D(s) } // SliceToArrayPointer; Deref | Convert
			func f10[S []rune, D *[2]rune | [2]rune](s S)                   { _ = D(s) } // SliceToArrayPointer | SliceToArrayPointer; Deref
			func f11[S []rune, D *[2]rune | [2]rune | string](s S)          { _ = D(s) } // SliceToArrayPointer | SliceToArrayPointer; Deref | Convert
			func f12[S []rune, D []rune](s S)                               { _ = D(s) } // ChangeType
			func f13[S []rune, D []rune | string](s S)                      { _ = D(s) } // Convert | ChangeType
			func f14[S []rune, D []rune | [2]rune](s S)                     { _ = D(s) } // ChangeType | SliceToArrayPointer; Deref
			func f15[S []rune, D []rune | [2]rune | string](s S)            { _ = D(s) } // ChangeType | SliceToArrayPointer; Deref | Convert
			func f16[S []rune, D []rune | *[2]rune](s S)                    { _ = D(s) } // ChangeType | SliceToArrayPointer
			func f17[S []rune, D []rune | *[2]rune | string](s S)           { _ = D(s) } // ChangeType | SliceToArrayPointer | Convert
			func f18[S []rune, D []rune | *[2]rune | [2]rune](s S)          { _ = D(s) } // ChangeType | SliceToArrayPointer | SliceToArrayPointer; Deref
			func f19[S []rune, D []rune | *[2]rune | [2]rune | string](s S) { _ = D(s) } // ChangeType | SliceToArrayPointer | SliceToArrayPointer; Deref | Convert
			func f20[S []rune | string, D string](s S)                      { _ = D(s) } // Convert | ChangeType
			func f21[S []rune | string, D []rune](s S)                      { _ = D(s) } // Convert | ChangeType
			func f22[S []rune | string, D []rune | string](s S)             { _ = D(s) } // ChangeType | Convert | Convert | ChangeType
			func f23[S []rune | [2]rune, D [2]rune](s S)                    { _ = D(s) } // SliceToArrayPointer; Deref | ChangeType
			func f24[S []rune | *[2]rune, D *[2]rune](s S)                  { _ = D(s) } // SliceToArrayPointer | ChangeType
			`, nil,
		},
		{
			"matching named and underlying types", `
			package p
			type a string
			type b string
			func g0[S []rune | a | b, D []rune | a | b](s S)      { _ = D(s) }
			func g1[S []rune | ~string, D []rune | a | b](s S)    { _ = D(s) }
			func g2[S []rune | a | b, D []rune | ~string](s S)    { _ = D(s) }
			func g3[S []rune | ~string, D []rune |~string](s S)   { _ = D(s) }
			`, nil,
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			fset := token.NewFileSet()
			f, err := parser.ParseFile(fset, "p.go", tc.src, parser.ParseComments)
			if err != nil {
				t.Error(err)
			}
			files := []*ast.File{f}

			pkg := types.NewPackage("p", "")
			conf := &types.Config{Importer: tc.importer}
			_, _, err = ssautil.BuildPackage(conf, fset, pkg, files, ssa.SanityCheckFunctions)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}
