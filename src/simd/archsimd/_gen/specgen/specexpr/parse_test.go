// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specexpr

import (
	"reflect"
	"strings"
	"testing"
)

func TestParseExpr(t *testing.T) {
	tests := []struct {
		expr string
		want Expr
	}{
		{
			expr: "10",
			want: Int(10),
		},
		{
			expr: "v",
			want: Variable("v"),
		},
		{
			expr: "v * 2",
			want: &BinExpr{OpTimes, Variable("v"), Int(2)},
		},
		{
			expr: "v / 2",
			want: &BinExpr{OpDiv, Variable("v"), Int(2)},
		},
		{
			expr: "x = y * 2",
			want: &BinExpr{OpEqual, Variable("x"), &BinExpr{OpTimes, Variable("y"), Int(2)}},
		},
		{
			expr: "a > b",
			want: &BinExpr{OpGreaterThan, Variable("a"), Variable("b")},
		},
		{
			expr: "a >= b",
			want: &BinExpr{OpGreaterOrEqual, Variable("a"), Variable("b")},
		},
		{
			expr: "a < b",
			want: &BinExpr{OpLessThan, Variable("a"), Variable("b")},
		},
		{
			expr: "a <= b",
			want: &BinExpr{OpLessOrEqual, Variable("a"), Variable("b")},
		},
		{
			expr: "(v * 2) / 3",
			want: &BinExpr{OpDiv, &BinExpr{OpTimes, Variable("v"), Int(2)}, Int(3)},
		},
		{
			expr: "Int32x4",
			want: makeVectorL(MakeBasic(&Literal{"int"}, Int(32)), Int(4)),
		},
		{
			expr: "Float64s",
			want: MakeVector(MakeBasic(&Literal{"float"}, Int(64)), VW()),
		},
		{
			expr: "{B}{N}x{L}",
			want: makeVectorL(MakeBasic(Variable("B"), Variable("N")), Variable("L")),
		},
		{
			expr: "{B}{N}w{W}",
			want: MakeVector(MakeBasic(Variable("B"), Variable("N")), Variable("W")),
		},
		{
			expr: "{xB}{xN*2}x{xL/2}",
			want: makeVectorL(
				MakeBasic(Variable("xB"), &BinExpr{OpTimes, Variable("xN"), Int(2)}),
				&BinExpr{OpDiv, Variable("xL"), Int(2)},
			),
		},
	}

	for _, tc := range tests {
		t.Run(tc.expr, func(t *testing.T) {
			got, err := ParseExpr(tc.expr)
			if err != nil {
				t.Fatalf("ParseExpr(%q) failed: %v", tc.expr, err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("ParseExpr(%q) = %+v; want %+v", tc.expr, got, tc.want)
			}
		})
	}
}

func TestParseExprErrors(t *testing.T) {
	tests := []struct {
		expr    string
		wantErr string
	}{
		{
			expr:    "",
			wantErr: "unexpected end",
		},
		{
			expr:    "12 34",
			wantErr: "unexpected trailing characters",
		},
		{
			expr:    "(12",
			wantErr: "expected ')'",
		},
		{
			expr:    "Int32x",
			wantErr: "expected number",
		},
		{
			expr:    "Int32w",
			wantErr: "expected number",
		},
		{
			expr:    "{B",
			wantErr: "'{' missing close '}' in symbolic shape at 1",
		},
		{
			expr:    "Int32x{}",
			wantErr: "unexpected character '}'",
		},
	}

	for _, tc := range tests {
		t.Run(tc.expr, func(t *testing.T) {
			_, err := ParseExpr(tc.expr)
			if err == nil {
				t.Fatalf("ParseExpr(%q) succeeded; want error containing %q", tc.expr, tc.wantErr)
			}
			if gotErr := err.Error(); !strings.Contains(gotErr, tc.wantErr) {
				t.Errorf("ParseExpr(%q) returned error %q; want error containing %q", tc.expr, gotErr, tc.wantErr)
			}
		})
	}
}
