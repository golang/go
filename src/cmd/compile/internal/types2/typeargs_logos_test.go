// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2_test

import (
	"fmt"
	"strings"
	"testing"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// An explicit type argument has the same meaning in every slot. The strict,
// alias, and later-slot controls provide an independent type identity oracle
// for the ambiguous first slot used by functions, methods, and type literals.
func TestLogosGenericTypeArgumentPositions(t *testing.T) {
	for _, typ := range []string{
		"int",
		"int?",
		"(int?)?",
		"*int?",
		"*(int?)",
		"(*int)?",
		"**int?",
		"[]int?",
		"<-chan int?",
		"(<-chan int)?",
		"(int, string)",
		"(int?, (string, bool))",
		"(x int, y string)",
		"(x: int, y: string)",
		"(x *int?, y *(int?))",
		"(x []int, y [2]string)",
		"(x map[string]int, y chan int)",
		"(x (int, string), y (int?)?)",
	} {
		t.Run(typ, func(t *testing.T) {
			for _, form := range []struct {
				name string
				src  string
			}{
				{"strict", "var _ unary[%s]"},
				{"alias", "type argument = %[1]s; var _ unary[%[1]s] = identity[argument]"},
				{"later", "var _ unary[%[1]s] = second[int, %[1]s]"},
				{"function", "var _ unary[%[1]s] = identity[%[1]s]"},
				{"method", "var _ unary[%[1]s] = receiver{}.identity[%[1]s]"},
				{"literal", "var _ box[%[1]s] = box[%[1]s]{}"},
			} {
				t.Run(form.name, func(t *testing.T) {
					src := genericTypeArgumentPrelude + fmt.Sprintf(form.src, typ) + "\n"
					if errs := checkGenericTypeArgumentSource(t, src); len(errs) != 0 {
						t.Fatalf("type %s in %s position:\n%s", typ, form.name, strings.Join(errs, "\n"))
					}
				})
			}
		})
	}
}

const genericTypeArgumentPrelude = `package p

type unary[T any] func(T) T
type box[T any] struct{}
type receiver struct{}
func identity[T any](v T) T { return v }
func second[A, B any](v B) B { return v }
func (receiver) identity[T any](v T) T { return v }
`

func TestLogosTypeArgumentIndexControls(t *testing.T) {
	const src = `package p

type key = (int, int)
func index() (int, error) { return 0, nil }
func use(m map[key]int, xs []int, optional int?) error {
	int, string := 1, 2
	_ = m[(int, string)]
	_ = xs[int + string - 1]
	_ = xs[optional ?? 0]
	_ = xs[0:1]
	_ = xs[0:1:2]
	%s
	return nil
}
`
	if errs := checkGenericTypeArgumentSource(t, fmt.Sprintf(src, "")); len(errs) != 0 {
		t.Fatalf("ordinary index expressions rejected:\n%s", strings.Join(errs, "\n"))
	}
	errs := checkGenericTypeArgumentSource(t, fmt.Sprintf(src, "_ = xs[index()?]"))
	if len(errs) != 1 || !strings.Contains(errs[0], "[correctness::misplaced_try]") {
		t.Fatalf("indexed propagation diagnostics = %q, want one misplaced_try", errs)
	}
}

func checkGenericTypeArgumentSource(t *testing.T, src string) (errs []string) {
	t.Helper()
	defer func() {
		if p := recover(); p != nil {
			errs = append(errs, fmt.Sprintf("checker panic: %v", p))
		}
	}()
	files, parseErrs := parseFiles(t, []string{"typeargs.lgo"}, [][]byte{[]byte(src)}, syntax.CheckBranches|syntax.LogosMode)
	for _, err := range parseErrs {
		errs = append(errs, err.Error())
	}
	if len(errs) != 0 {
		return errs
	}
	conf := types2.Config{GoVersion: "go1.27", Logos: true, Error: func(err error) {
		if !types2.LogosWarnOnly(err) {
			errs = append(errs, err.Error())
		}
	}}
	_, _ = conf.Check("p", files, nil)
	return errs
}
