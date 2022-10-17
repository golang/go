// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"testing"

	"golang.org/x/tools/internal/testenv"

	. "golang.org/x/tools/gopls/internal/lsp/regtest"
)

func TestStaticcheckGenerics(t *testing.T) {
	testenv.NeedsGo1Point(t, 18) // generics were introduced in Go 1.18

	const files = `
-- go.mod --
module mod.com

go 1.18
-- a/a.go --
package a

import (
	"errors"
	"sort"
	"strings"
)

func Zero[P any]() P {
	var p P
	return p
}

type Inst[P any] struct {
	Field P
}

func testGenerics[P *T, T any](p P) {
	// Calls to instantiated functions should not break checks.
	slice := Zero[string]()
	sort.Slice(slice, func(i, j int) bool {
		return slice[i] < slice[j]
	})

	// Usage of instantiated fields should not break checks.
	g := Inst[string]{"hello"}
	g.Field = strings.TrimLeft(g.Field, "12234")

	// Use of type parameters should not break checks.
	var q P
	p = q // SA4009: p is overwritten before its first use
	q = &*p // SA4001: &* will be simplified
}


// FooErr should be called ErrFoo (ST1012)
var FooErr error = errors.New("foo")
`

	WithOptions(
		Settings{"staticcheck": true},
	).Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("a/a.go")
		env.Await(
			env.DiagnosticAtRegexpFromSource("a/a.go", "sort.Slice", "sortslice"),
			env.DiagnosticAtRegexpFromSource("a/a.go", "sort.Slice.(slice)", "SA1028"),
			env.DiagnosticAtRegexpFromSource("a/a.go", "var (FooErr)", "ST1012"),
			env.DiagnosticAtRegexpFromSource("a/a.go", `"12234"`, "SA1024"),
			env.DiagnosticAtRegexpFromSource("a/a.go", "testGenerics.*(p P)", "SA4009"),
			env.DiagnosticAtRegexpFromSource("a/a.go", "q = (&\\*p)", "SA4001"),
		)
	})
}

// Test for golang/go#56270: an analysis with related info should not panic if
// analysis.RelatedInformation.End is not set.
func TestStaticcheckRelatedInfo(t *testing.T) {
	testenv.NeedsGo1Point(t, 17) // staticcheck is only supported at Go 1.17+
	const files = `
-- go.mod --
module mod.test

go 1.18
-- p.go --
package p

import (
	"fmt"
)

func Foo(enabled interface{}) {
	if enabled, ok := enabled.(bool); ok {
	} else {
		_ = fmt.Sprintf("invalid type %T", enabled) // enabled is always bool here
	}
}
`

	WithOptions(
		Settings{"staticcheck": true},
	).Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("p.go")
		env.Await(
			OnceMet(
				env.DoneWithOpen(),
				env.DiagnosticAtRegexpFromSource("p.go", ", (enabled)", "SA9008"),
			),
		)
	})
}
