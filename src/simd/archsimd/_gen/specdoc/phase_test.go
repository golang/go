// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specdoc

import (
	"errors"
	"slices"
	"strings"
	"testing"

	"simd/archsimd/_gen/specgen"
	"simd/archsimd/_gen/specgen/specexpr"
)

func TestRejectUnknownPhase(t *testing.T) {
	vec := func(name string) specexpr.Type {
		typ, err := specexpr.ParseTypeName(name)
		if err != nil {
			t.Fatal(err)
		}
		return typ
	}
	method := func(recv, name string, experiment bool) *specgen.Func {
		return &specgen.Func{Name: name, Recv: specgen.Arg{Name: "x", Type: vec(recv)}, Experiment: experiment}
	}
	idx := specgen.NewIndex([]*specgen.Func{
		method("Int32x4", "Add", false),
		method("Int32x4", "Exp", true),
		method("Int8s", "Add", false),
	}, []*specgen.TypeDecl{
		typeDecl(t, "Int32x4", "Int32x4 is a vector."),
		typeDecl(t, "Int8s", "Int8s is a scalable vector."),
	})

	const decls = `
type Int32x4 struct{}
type Int8s struct{}
type X86Features struct{}
type Other struct{}
type unexported struct{}

var X86 X86Features
var Var int
const Const = 1

func (Int32x4) Add(y Int32x4) Int32x4
func (Int32x4) Exp() Int32x4
func (Int32x4) Unknown() Int32x4

// Deprecated: use Add. Spec doesn't describe deprecated APIs.
func (Int32x4) Sub(y Int32x4) Int32x4

func (Int8s) Add(y Int8s) Int8s
func (X86Features) AVX() bool
func (X86Features) SHA() bool
func (unexported) Unknown()
func unknown()
`
	tests := []struct {
		name     string
		filename string
		header   string
		want     []string // violating declarations
	}{
		{
			name:   "stable archsimd",
			header: "//go:build goexperiment.simdstable && amd64\n\npackage archsimd\n",
			want:   []string{"(Int32x4) Exp", "(Int32x4) Sub", "(Int32x4) Unknown", "(Int8s) Add", "Const", "Int8s", "Other", "Var"},
		},
		{
			// archsimd's API is generally available only for the
			// architectures in the stable archsimd architectures, so a file for
			// any architecture can't declare it...
			name:   "stable archsimd, any architecture",
			header: "//go:build goexperiment.simdstable\n\npackage archsimd\n",
			want:   []string{"(Int32x4) Add", "(Int32x4) Exp", "(Int32x4) Sub", "(Int32x4) Unknown", "(Int8s) Add", "Const", "Int32x4", "Int8s", "Other", "Var"},
		},
		{
			// ... and neither can a file for another architecture.
			name:     "stable archsimd, riscv64",
			filename: "simd/archsimd/ops_riscv64.go",
			header:   "//go:build goexperiment.simdstable\n\npackage archsimd\n",
			want:     []string{"(Int32x4) Add", "(Int32x4) Exp", "(Int32x4) Sub", "(Int32x4) Unknown", "(Int8s) Add", "Const", "Int32x4", "Int8s", "Other", "Var"},
		},
		{
			// ... or a file for a GA architecture or another one.
			name:   "stable archsimd, amd64 or mips",
			header: "//go:build goexperiment.simdstable && (amd64 || mips)\n\npackage archsimd\n",
			want:   []string{"(Int32x4) Add", "(Int32x4) Exp", "(Int32x4) Sub", "(Int32x4) Unknown", "(Int8s) Add", "Const", "Int32x4", "Int8s", "Other", "Var"},
		},
		{
			name:   "experimental archsimd",
			header: "//go:build goexperiment.simd && amd64\n\npackage archsimd\n",
			want:   []string{"(Int32x4) Add", "Int32x4"},
		},
		{
			// On an architecture whose archsimd API isn't generally
			// available, everything is experimental.
			name:     "experimental archsimd, riscv64",
			filename: "simd/archsimd/ops_riscv64.go",
			header:   "//go:build goexperiment.simd\n\npackage archsimd\n",
		},
		{
			// The architecture rule is only for archsimd, and the non-spec
			// APIs depend on the package.
			name:   "stable simd",
			header: "//go:build goexperiment.simdstable\n\npackage simd\n",
			want:   []string{"(Int32x4) Exp", "(Int32x4) Sub", "(Int32x4) Unknown", "(X86Features) AVX", "(X86Features) SHA", "Const", "Other", "Var", "X86", "X86Features"},
		},
		{
			// In package simd, scalable types aren't SVE.
			name:   "experimental simd",
			header: "//go:build goexperiment.simd\n\npackage simd\n",
			want:   []string{"(Int32x4) Add", "(Int8s) Add", "Int32x4", "Int8s"},
		},
		{
			// A file without a phase builds regardless of GOEXPERIMENT, so
			// it can't export anything.
			name:   "no phase",
			header: "//go:build amd64\n\npackage archsimd\n",
			want:   []string{"(Int32x4) Add", "(Int32x4) Exp", "(Int32x4) Sub", "(Int32x4) Unknown", "(Int8s) Add", "(X86Features) AVX", "(X86Features) SHA", "Const", "Int32x4", "Int8s", "Other", "Var", "X86", "X86Features"},
		},
		{
			name:     "test file",
			filename: "x_test.go",
			header:   "//go:build goexperiment.simdstable\n\npackage archsimd\n",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Fill([]byte(tt.header+decls), idx, Options{
				Filename:            tt.filename,
				AllowDocRewrite:     true,
				AllowNameMismatches: true,
				NoFillDoc:           true,
				NoFillNames:         true,
				RejectUnknown:       true,
			})
			var got []string
			var rep *Report
			if errors.As(err, &rep) {
				for _, v := range rep.UnknownDecls {
					got = append(got, v.D.String())
				}
				for _, v := range rep.PhaseViolations {
					got = append(got, v.D.String())
				}
			} else if err != nil {
				t.Fatal(err)
			}
			slices.Sort(got)
			if !slices.Equal(got, tt.want) {
				t.Errorf("violations:\n\t%s\nwant:\n\t%s", strings.Join(got, "\n\t"), strings.Join(tt.want, "\n\t"))
			}
		})
	}

	// Without RejectUnknown, nothing is rejected.
	_, err := Fill([]byte("//go:build goexperiment.simdstable\n\npackage archsimd\n"+decls), idx, Options{AllowDocRewrite: true, AllowNameMismatches: true})
	var rep *Report
	if errors.As(err, &rep) && (len(rep.PhaseViolations) > 0 || len(rep.UnknownDecls) > 0) {
		t.Errorf("rejected declarations without RejectUnknown: %v", rep)
	}

	// A file that never builds is an error.
	_, err = Fill([]byte("//go:build goexperiment.simd && !goexperiment.simdstable\n\npackage archsimd\n"), idx, Options{RejectUnknown: true})
	if err == nil || errors.As(err, &rep) {
		t.Errorf("want error for a file that never builds, got %v", err)
	}
}
