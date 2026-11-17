// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"internal/testenv"
	"strings"
	"testing"
)

const arrayIndexPreamble = `package p

type token uint8

const (
	tokenLight token = iota
	tokenDark
	tokenCount
)

var names [2]string

func validate(t token) string {
	if t >= tokenCount { panic("invalid token") }
	return ""
}

`

func TestArrayIndexOutOfBounds(t *testing.T) {
	cases := []struct {
		name string
		body string
	}{
		{
			name: "guardedReceiver",
			body: `func (t token) String() string {
		names := [...]string{"light", "dark"}
		if t >= tokenCount { return "unknown" }
		return names[t]
	}`,
		},
		{
			name: "offByOneReceiver",
			body: `func (t token) String() string {
		names := [...]string{"light", "dark"}
		if t > tokenCount { return "unknown" }
		return names[t] // want
	}`,
		},
		{
			name: "signedBothBounds",
			body: `func f(i int8) string {
		names := [...]string{"light", "dark"}
		if i < 0 || i >= 2 { return "unknown" }
		return names[i]
	}`,
		},
		{
			name: "signedMissingLowerBound",
			body: `func f(i int8) string {
		names := [...]string{"light", "dark"}
		if i >= 2 { return "unknown" }
		return names[i] // want
	}`,
		},
		{
			name: "constantOnLeftSafe",
			body: `func f(t token) string {
		if tokenCount <= t { return "unknown" }
		return names[t]
	}`,
		},
		{
			name: "constantOnLeftOffByOne",
			body: `func f(t token) string {
		if tokenCount < t { return "unknown" }
		return names[t] // want
	}`,
		},
		{
			name: "inclusiveSafe",
			body: `func f(t token) string {
		if t > tokenDark { return "unknown" }
		return names[t]
	}`,
		},
		{
			name: "trueBranchOffByOne",
			body: `func f(t token) string {
		if t <= tokenCount {
			return names[t] // want
		}
		return "unknown"
	}`,
		},
		{
			name: "negatedGuard",
			body: `func f(t token) string {
		if !(t < tokenCount) { return "unknown" }
		return names[t]
	}`,
		},
		{
			name: "compoundBoundary",
			body: `func f(t token) string {
		if t == tokenLight || t == tokenCount {
			return names[t] // want
		}
		return "unknown"
	}`,
		},
		{
			name: "unreachableBoundary",
			body: `func f(t token) string {
		if t < tokenCount && t == tokenCount { return names[t] }
		return "unknown"
	}`,
		},
		{
			name: "disjointGuard",
			body: `func f(t token) string {
		if t != tokenCount {
			if t <= tokenCount { return names[t] }
		}
		return "unknown"
	}`,
		},
		{
			name: "equalityBoundary",
			body: `func f(t token) string {
		if t == tokenCount {
			return names[t] // want
		}
		return "unknown"
	}`,
		},
		{
			name: "branchRejoins",
			body: `func f(t token) string {
		if t >= tokenCount {}
		return names[t] // want
	}`,
		},
		{
			name: "earlyPureReturn",
			body: `func f(t token) string {
		if true { return "unknown" }
		return names[t]
	}`,
		},
		{
			name: "signedReportsOnce",
			body: `func f(i int8) string {
		return names[i] // want
	}`,
		},
		{
			name: "arrayCoversUnsignedType",
			body: `func f(t token) string {
		var all [256]string
		return all[t]
	}`,
		},
		{
			name: "signedUpperWitnessUnrepresentable",
			body: `func f(i int8) string {
		var all [128]string
		if i < 0 { return "unknown" }
		return all[i]
	}`,
		},
		{
			name: "uint64Parameter",
			body: `func f(i uint64) string {
		if i > uint64(2) { return "unknown" }
		return names[i] // want
	}`,
		},
		{
			name: "nativeIntParameter",
			body: `func f(i int) string {
		if i < 0 || i > 2 { return "unknown" }
		return names[i] // want
	}`,
		},
		{
			name: "nativeUintParameter",
			body: `func f(i uint) string {
		if i > 2 { return "unknown" }
		return names[i] // want
	}`,
		},
		{
			name: "unknownGuard",
			body: `func f(t token, valid bool) string {
		if valid { return names[t] }
		return "unknown"
	}`,
		},
		{
			name: "callBeforeAccess",
			body: `func f(t token) string {
		prefix := validate(t)
		return prefix + names[t]
	}`,
		},
		{
			name: "callAfterAccessInBinary",
			body: `func f(t token) string { return names[t] + validate(t) }`,
		},
		{
			name: "callAfterAccessInTuple",
			body: `func f(t token) (string, string) { return names[t], validate(t) }`,
		},
		{
			name: "divisionBeforeAccess",
			body: `func f(t token, divisor int) (string, int) {
		value := 1 / divisor
		return names[t], value
	}`,
		},
		{
			name: "dereferenceBeforeAccess",
			body: `func f(t token, pointer *int) (string, int) {
		value := *pointer
		return names[t], value
	}`,
		},
		{
			name: "sliceAccessBeforeArray",
			body: `func f(t token, values []string, i int) (string, string) {
		value := values[i]
		return names[t], value
	}`,
		},
		{
			name: "shiftBeforeAccess",
			body: `func f(t token, shift int) (string, int) {
		value := 1 << shift
		return names[t], value
	}`,
		},
		{
			name: "assertionBeforeAccess",
			body: `func f(t token, input any) (string, string) {
		value := input.(string)
		return names[t], value
	}`,
		},
		{
			name: "interfaceComparisonBeforeAccess",
			body: `var bad any = []int{1}
	func f(t token) string {
		equal := bad == bad
		_ = equal
		return names[t]
	}`,
		},
		{
			name: "assignedParameter",
			body: `func f(t token) string { t = tokenDark; return names[t] }`,
		},
		{
			name: "incrementedParameter",
			body: `func f(t token) string { t++; return names[t] }`,
		},
		{
			name: "unreachableParameterWrite",
			body: `func f(t token) string {
		if false { t = tokenLight }
		return names[t]
	}`,
		},
		{
			name: "addressTaken",
			body: `func f(t token) string { _ = &t; return names[t] }`,
		},
		{
			name: "capturedParameter",
			body: `func f(t token) string {
		captured := func() token { return t }
		_ = captured
		return names[t]
	}`,
		},
		{
			name: "shadowedName",
			body: `func f(t token) string {
		if t >= tokenCount { t := tokenDark; return names[t] }
		return names[t]
	}`,
		},
		{
			name: "loopBeforeAccess",
			body: `func f(t token) string { for range 1 {}; return names[t] }`,
		},
		{
			name: "jumpBeforeAccess",
			body: `func f(t token) string { goto access; access: return names[t] }`,
		},
		{
			name: "pointerArray",
			body: `func f(t token, pointer *[2]string) string { return pointer[t] }`,
		},
		{
			name: "slice",
			body: `func f(t token, values []string) string { return values[t] }`,
		},
		{
			name: "convertedIndex",
			body: `func f(t token) string { return names[int(t)] }`,
		},
		{
			name: "arithmeticIndex",
			body: `func f(t token) string { return names[t+1] }`,
		},
		{
			name: "localIndexAlias",
			body: `func f(t token) string { i := t; return names[i] }`,
		},
		{
			name: "typeParameterIndex",
			body: `func f[T ~uint8](i T) string { return names[i] }`,
		},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			for name, kind := range map[string]Kind{"x.go": KindGo, "x.lgo": KindLgo} {
				t.Run(name, func(t *testing.T) {
					expect(t, kind, nil, testSource{name, arrayIndexPreamble + tt.body}, finding{"correctness::array_index_out_of_bounds", Deny})
				})
			}
		})
	}
}

func TestArrayIndexOutOfBoundsDiagnostic(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	cases := []struct {
		name    string
		body    string
		witness string
		operand string
	}{
		{
			name: "upperBoundary",
			body: `func f(t token) string {
		if t > tokenCount { return "unknown" }
		return names[t]
	}`,
			witness: "can be 2 at this access",
			operand: "t",
		},
		{
			name: "negativeBoundary",
			body: `func f(i int8) string {
		if i >= 2 { return "unknown" }
		return names[i]
	}`,
			witness: "can be -1 at this access",
			operand: "i",
		},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			for name, kind := range map[string]Kind{"x.go": KindGo, "x.lgo": KindLgo} {
				t.Run(name, func(t *testing.T) {
					src := arrayIndexPreamble + tt.body
					results := Run(check(t, kind, name, src))
					if len(results) != 1 {
						t.Fatalf("findings = %d, want 1: %+v", len(results), results)
					}
					got := results[0]
					if got.Rule != "correctness::array_index_out_of_bounds" || got.Severity != Deny {
						t.Errorf("finding = %s %s, want correctness::array_index_out_of_bounds deny", got.Rule, got.Severity)
					}
					if !strings.Contains(got.Message, tt.witness) || !strings.Contains(got.Message, "bounds [0, 2)") {
						t.Errorf("message = %q, want %q and array bounds [0, 2)", got.Message, tt.witness)
					}
					if span := spanText(src, got); span != tt.operand {
						t.Errorf("span = %q, want %q", span, tt.operand)
					}
				})
			}
		})
	}
}

func TestArrayIndexOutOfBoundsAllow(t *testing.T) {
	const body = `//logos:allow(correctness::array_index_out_of_bounds)
func f(t token) string { return names[t] }
`
	for name, kind := range map[string]Kind{"x.go": KindGo, "x.lgo": KindLgo} {
		t.Run(name, func(t *testing.T) {
			expect(t, kind, nil, testSource{name, arrayIndexPreamble + body}, finding{"correctness::array_index_out_of_bounds", Deny})
		})
	}
}
