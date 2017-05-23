// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package constant

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
			continue
		}

		if x0 != nil && !eql(x, x0) {
			t.Errorf("%s: x changed to %s", test, x)
			continue
		}

		if !eql(y, y0) {
			t.Errorf("%s: y changed to %s", test, y)
			continue
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
// String tests

var xxx = strings.Repeat("x", 68)
var issue14262 = `"بموجب الشروط التالية نسب المصنف — يجب عليك أن تنسب العمل بالطريقة التي تحددها المؤلف أو المرخص (ولكن ليس بأي حال من الأحوال أن توحي وتقترح بتحول أو استخدامك للعمل).  المشاركة على قدم المساواة — إذا كنت يعدل ، والتغيير ، أو الاستفادة من هذا العمل ، قد ينتج عن توزيع العمل إلا في ظل تشابه او تطابق فى واحد لهذا الترخيص."`

var stringTests = []struct {
	input, short, exact string
}{
	// Unknown
	{"", "unknown", "unknown"},
	{"0x", "unknown", "unknown"},
	{"'", "unknown", "unknown"},
	{"1f0", "unknown", "unknown"},
	{"unknown", "unknown", "unknown"},

	// Bool
	{"true", "true", "true"},
	{"false", "false", "false"},

	// String
	{`""`, `""`, `""`},
	{`"foo"`, `"foo"`, `"foo"`},
	{`"` + xxx + `xx"`, `"` + xxx + `xx"`, `"` + xxx + `xx"`},
	{`"` + xxx + `xxx"`, `"` + xxx + `...`, `"` + xxx + `xxx"`},
	{`"` + xxx + xxx + `xxx"`, `"` + xxx + `...`, `"` + xxx + xxx + `xxx"`},
	{issue14262, `"بموجب الشروط التالية نسب المصنف — يجب عليك أن تنسب العمل بالطريقة ال...`, issue14262},

	// Int
	{"0", "0", "0"},
	{"-1", "-1", "-1"},
	{"12345", "12345", "12345"},
	{"-12345678901234567890", "-12345678901234567890", "-12345678901234567890"},
	{"12345678901234567890", "12345678901234567890", "12345678901234567890"},

	// Float
	{"0.", "0", "0"},
	{"-0.0", "0", "0"},
	{"10.0", "10", "10"},
	{"2.1", "2.1", "21/10"},
	{"-2.1", "-2.1", "-21/10"},
	{"1e9999", "1e+9999", "0x.f8d4a9da224650a8cb2959e10d985ad92adbd44c62917e608b1f24c0e1b76b6f61edffeb15c135a4b601637315f7662f325f82325422b244286a07663c9415d2p+33216"},
	{"1e-9999", "1e-9999", "0x.83b01ba6d8c0425eec1b21e96f7742d63c2653ed0a024cf8a2f9686df578d7b07d7a83d84df6a2ec70a921d1f6cd5574893a7eda4d28ee719e13a5dce2700759p-33215"},
	{"2.71828182845904523536028747135266249775724709369995957496696763", "2.71828", "271828182845904523536028747135266249775724709369995957496696763/100000000000000000000000000000000000000000000000000000000000000"},
	{"0e9999999999", "0", "0"}, // issue #16176

	// Complex
	{"0i", "(0 + 0i)", "(0 + 0i)"},
	{"-0i", "(0 + 0i)", "(0 + 0i)"},
	{"10i", "(0 + 10i)", "(0 + 10i)"},
	{"-10i", "(0 + -10i)", "(0 + -10i)"},
	{"1e9999i", "(0 + 1e+9999i)", "(0 + 0x.f8d4a9da224650a8cb2959e10d985ad92adbd44c62917e608b1f24c0e1b76b6f61edffeb15c135a4b601637315f7662f325f82325422b244286a07663c9415d2p+33216i)"},
}

func TestString(t *testing.T) {
	for _, test := range stringTests {
		x := val(test.input)
		if got := x.String(); got != test.short {
			t.Errorf("%s: got %q; want %q as short string", test.input, got, test.short)
		}
		if got := x.ExactString(); got != test.exact {
			t.Errorf("%s: got %q; want %q as exact string", test.input, got, test.exact)
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

	if i := strings.IndexByte(lit, '/'); i >= 0 {
		// assume fraction
		a := MakeFromLiteral(lit[:i], token.INT, 0)
		b := MakeFromLiteral(lit[i+1:], token.INT, 0)
		return BinaryOp(a, token.QUO, b)
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

	return MakeFromLiteral(lit, tok, 0)
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
		return UnaryOp(op, y, 0)
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
	"0",
	"1",
	"-1",
	"1.2",
	"-0.991",
	"2.718281828",
	"3.14159265358979323e-10",
	"1e100",
	"1e1000",
}

func TestFractions(t *testing.T) {
	for _, test := range fracTests {
		x := val(test)
		// We don't check the actual numerator and denominator because they
		// are unlikely to be 100% correct due to floatVal rounding errors.
		// Instead, we compute the fraction again and compare the rounded
		// result.
		q := BinaryOp(Num(x), token.QUO, Denom(x))
		got := q.String()
		want := x.String()
		if got != want {
			t.Errorf("%s: got quotient %s, want %s", x, got, want)
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

func TestUnknown(t *testing.T) {
	u := MakeUnknown()
	var values = []Value{
		u,
		MakeBool(false), // token.ADD ok below, operation is never considered
		MakeString(""),
		MakeInt64(1),
		MakeFromLiteral("-1234567890123456789012345678901234567890", token.INT, 0),
		MakeFloat64(1.2),
		MakeImag(MakeFloat64(1.2)),
	}
	for _, val := range values {
		x, y := val, u
		for i := range [2]int{} {
			if i == 1 {
				x, y = y, x
			}
			if got := BinaryOp(x, token.ADD, y); got.Kind() != Unknown {
				t.Errorf("%s + %s: got %s; want %s", x, y, got, u)
			}
			if got := Compare(x, token.EQL, y); got {
				t.Errorf("%s == %s: got true; want false", x, y)
			}
		}
	}
}
