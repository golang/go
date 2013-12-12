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

var opTests = []string{
	// unary operations
	`+ 0 = 0`,
	`+ ? = ?`,
	`- 1 = -1`,
	`- ? = ?`,
	`^ 0 = -1`,
	`^ ? = ?`,

	`! true = false`,
	`! false = true`,
	`! ? = ?`,

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
	`? + 0 = ?`,
	`0 + ? = ?`,

	`0 - 0 = 0`,
	`0 - 0.1 = -0.1`,
	`0 - 0.1i = -0.1i`,
	`1e100 - 1e100 = 0`,
	`? - 0 = ?`,
	`0 - ? = ?`,

	`0 * 0 = 0`,
	`1 * 0.1 = 0.1`,
	`1 * 0.1i = 0.1i`,
	`1i * 1i = -1`,
	`? * 0 = ?`,
	`0 * ? = ?`,

	`0 / 0 = "division_by_zero"`,
	`10 / 2 = 5`,
	`5 / 3 = 5/3`,
	`5i / 3i = 5/3`,
	`? / 0 = ?`,
	`0 / ? = ?`,

	`0 % 0 = "runtime_error:_integer_divide_by_zero"`, // TODO(gri) should be the same as for /
	`10 % 3 = 1`,
	`? % 0 = ?`,
	`0 % ? = ?`,

	`0 & 0 = 0`,
	`12345 & 0 = 0`,
	`0xff & 0xf = 0xf`,
	`? & 0 = ?`,
	`0 & ? = ?`,

	`0 | 0 = 0`,
	`12345 | 0 = 12345`,
	`0xb | 0xa0 = 0xab`,
	`? | 0 = ?`,
	`0 | ? = ?`,

	`0 ^ 0 = 0`,
	`1 ^ -1 = -2`,
	`? ^ 0 = ?`,
	`0 ^ ? = ?`,

	`0 &^ 0 = 0`,
	`0xf &^ 1 = 0xe`,
	`1 &^ 0xf = 0`,
	// etc.

	// shifts
	`0 << 0 = 0`,
	`1 << 10 = 1024`,
	`0 >> 0 = 0`,
	`1024 >> 10 == 1`,
	`? << 0 == ?`,
	`? >> 10 == ?`,
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

	`0 == 0 = true`,
	`0 != 0 = false`,
	`0 < 10 = true`,
	`10 <= 10 = true`,
	`0 > 10 = false`,
	`10 >= 10 = true`,

	`1/123456789 == 1/123456789 == true`,
	`1/123456789 != 1/123456789 == false`,
	`1/123456789 < 1/123456788 == true`,
	`1/123456788 <= 1/123456789 == false`,
	`0.11 > 0.11 = false`,
	`0.11 >= 0.11 = true`,

	`? == 0 = false`,
	`? != 0 = false`,
	`? < 10 = false`,
	`? <= 10 = false`,
	`? > 10 = false`,
	`? >= 10 = false`,

	`0 == ? = false`,
	`0 != ? = false`,
	`0 < ? = false`,
	`10 <= ? = false`,
	`0 > ? = false`,
	`10 >= ? = false`,

	// etc.
}

func TestOps(t *testing.T) {
	for _, test := range opTests {
		a := strings.Split(test, " ")
		i := 0 // operator index

		var x, x0 Value
		switch len(a) {
		case 4:
			// unary operation
		case 5:
			// binary operation
			x, x0 = val(a[0]), val(a[0])
			i = 1
		default:
			t.Errorf("invalid test case: %s", test)
			continue
		}

		op, ok := optab[a[i]]
		if !ok {
			panic("missing optab entry for " + a[i])
		}

		y, y0 := val(a[i+1]), val(a[i+1])

		got := doOp(x, op, y)
		want := val(a[i+3])
		if !eql(got, want) {
			t.Errorf("%s: got %s; want %s", test, got, want)
		}
		if x0 != nil && !eql(x, x0) {
			t.Errorf("%s: x changed to %s", test, x)
		}
		if !eql(y, y0) {
			t.Errorf("%s: y changed to %s", test, y)
		}
	}
}

func eql(x, y Value) bool {
	_, ux := x.(unknownVal)
	_, uy := y.(unknownVal)
	if ux || uy {
		return ux == uy
	}
	return Compare(x, token.EQL, y)
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

	tok := token.INT
	switch first, last := lit[0], lit[len(lit)-1]; {
	case first == '"' || first == '`':
		tok = token.STRING
		lit = strings.Replace(lit, "_", " ", -1)
	case first == '\'':
		tok = token.CHAR
	case last == 'i':
		tok = token.IMAG
	default:
		if !strings.HasPrefix(lit, "0x") && strings.ContainsAny(lit, "./Ee") {
			tok = token.FLOAT
		}
	}

	return MakeFromLiteral(lit, tok)
}

var optab = map[string]token.Token{
	"!": token.NOT,

	"+": token.ADD,
	"-": token.SUB,
	"*": token.MUL,
	"/": token.QUO,
	"%": token.REM,

	"<<": token.SHL,
	">>": token.SHR,

	"&":  token.AND,
	"|":  token.OR,
	"^":  token.XOR,
	"&^": token.AND_NOT,

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

// ----------------------------------------------------------------------------
// Other tests

var fracTests = []string{
	"0 0 1",
	"1 1 1",
	"-1 -1 1",
	"1.2 6 5",
	"-0.991 -991 1000",
	"1e100 1e100 1",
}

func TestFractions(t *testing.T) {
	for _, test := range fracTests {
		a := strings.Split(test, " ")
		if len(a) != 3 {
			t.Errorf("invalid test case: %s", test)
			continue
		}

		x := val(a[0])
		n := val(a[1])
		d := val(a[2])

		if got := Num(x); !eql(got, n) {
			t.Errorf("%s: got num = %s; want %s", test, got, n)
		}

		if got := Denom(x); !eql(got, d) {
			t.Errorf("%s: got denom = %s; want %s", test, got, d)
		}
	}
}

var bytesTests = []string{
	"0",
	"1",
	"123456789",
	"123456789012345678901234567890123456789012345678901234567890",
}

func TestBytes(t *testing.T) {
	for _, test := range bytesTests {
		x := val(test)
		bytes := Bytes(x)

		// special case 0
		if Sign(x) == 0 && len(bytes) != 0 {
			t.Errorf("%s: got %v; want empty byte slice", test, bytes)
		}

		if n := len(bytes); n > 0 && bytes[n-1] == 0 {
			t.Errorf("%s: got %v; want no leading 0 byte", test, bytes)
		}

		if got := MakeFromBytes(bytes); !eql(got, x) {
			t.Errorf("%s: got %s; want %s (bytes = %v)", test, got, x, bytes)
		}
	}
}
