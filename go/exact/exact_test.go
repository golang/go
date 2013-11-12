// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package exact

import (
	"go/token"
	"strings"
	"testing"
)

// TODO(gri) expand this test framework

var tests = []string{
	// unary operations
	`+ 0 = 0`,
	`- 1 = -1`,

	`! true = false`,
	`! false = true`,
	// etc.

	// binary operations
	`"" + "" = ""`,
	`"foo" + "" = "foo"`,
	`"" + "bar" = "bar"`,
	`"foo" + "bar" = "foobar"`,

	`0 + 0 = 0`,
	`0 + 0.1 = 0.1`,
	`0 + 0.1i = 0.1i`,
	`0.1 + 0.9 = 1`,
	`1e100 + 1e100 = 2e100`,

	`0 - 0 = 0`,
	`0 - 0.1 = -0.1`,
	`0 - 0.1i = -0.1i`,
	`1e100 - 1e100 = 0`,

	`0 * 0 = 0`,
	`1 * 0.1 = 0.1`,
	`1 * 0.1i = 0.1i`,
	`1i * 1i = -1`,

	`0 / 0 = "division_by_zero"`,
	`10 / 2 = 5`,
	`5 / 3 = 5/3`,
	`5i / 3i = 5/3`,

	`0 % 0 = "runtime_error:_integer_divide_by_zero"`, // TODO(gri) should be the same as for /
	`10 % 3 = 1`,
	// etc.

	// shifts
	`0 << 0 = 0`,
	`1 << 10 = 1024`,
	// etc.

	// comparisons
	`false == false = true`,
	`false == true = false`,
	`true == false = false`,
	`true == true = true`,

	`false != false = false`,
	`false != true = true`,
	`true != false = true`,
	`true != true = false`,

	`"foo" == "bar" = false`,
	`"foo" != "bar" = true`,
	`"foo" < "bar" = false`,
	`"foo" <= "bar" = false`,
	`"foo" > "bar" = true`,
	`"foo" >= "bar" = true`,

	`0 != 0 = false`,

	// etc.
}

func TestOps(t *testing.T) {
	for _, test := range tests {
		var got, want Value

		switch a := strings.Split(test, " "); len(a) {
		case 4:
			x := val(a[1])
			got = doOp(nil, op[a[0]], x)
			want = val(a[3])
			if !Compare(x, token.EQL, val(a[1])) {
				t.Errorf("%s failed: x changed to %s", test, x)
			}
		case 5:
			x := val(a[0])
			y := val(a[2])
			got = doOp(x, op[a[1]], y)
			want = val(a[4])
			if !Compare(x, token.EQL, val(a[0])) {
				t.Errorf("%s failed: x changed to %s", test, x)
			}
			if !Compare(y, token.EQL, val(a[2])) {
				t.Errorf("%s failed: y changed to %s", test, y)
			}
		default:
			t.Errorf("invalid test case: %s", test)
			continue
		}

		if !Compare(got, token.EQL, want) {
			t.Errorf("%s failed: got %s; want %s", test, got, want)
		}
	}
}

// ----------------------------------------------------------------------------
// Support functions

func val(lit string) Value {
	if len(lit) == 0 {
		return MakeUnknown()
	}

	switch lit {
	case "?":
		return MakeUnknown()
	case "true":
		return MakeBool(true)
	case "false":
		return MakeBool(false)
	}

	tok := token.FLOAT
	switch first, last := lit[0], lit[len(lit)-1]; {
	case first == '"' || first == '`':
		tok = token.STRING
		lit = strings.Replace(lit, "_", " ", -1)
	case first == '\'':
		tok = token.CHAR
	case last == 'i':
		tok = token.IMAG
	}

	return MakeFromLiteral(lit, tok)
}

var op = map[string]token.Token{
	"!": token.NOT,

	"+": token.ADD,
	"-": token.SUB,
	"*": token.MUL,
	"/": token.QUO,
	"%": token.REM,

	"<<": token.SHL,
	">>": token.SHR,

	"==": token.EQL,
	"!=": token.NEQ,
	"<":  token.LSS,
	"<=": token.LEQ,
	">":  token.GTR,
	">=": token.GEQ,
}

func panicHandler(v *Value) {
	switch p := recover().(type) {
	case nil:
		// nothing to do
	case string:
		*v = MakeString(p)
	case error:
		*v = MakeString(p.Error())
	default:
		panic(p)
	}
}

func doOp(x Value, op token.Token, y Value) (z Value) {
	defer panicHandler(&z)

	if x == nil {
		return UnaryOp(op, y, -1)
	}

	switch op {
	case token.EQL, token.NEQ, token.LSS, token.LEQ, token.GTR, token.GEQ:
		return MakeBool(Compare(x, op, y))
	case token.SHL, token.SHR:
		s, _ := Int64Val(y)
		return Shift(x, op, uint(s))
	default:
		return BinaryOp(x, op, y)
	}
}
