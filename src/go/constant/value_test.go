// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package constant

import (
	"fmt"
	"go/token"
	"math"
	"math/big"
	"strings"
	"testing"
)

var intTests = []string{
	// 0-octals
	`0_123 = 0123`,
	`0123_456 = 0123456`,

	// decimals
	`1_234 = 1234`,
	`1_234_567 = 1234567`,

	// hexadecimals
	`0X_0 = 0`,
	`0X_1234 = 0x1234`,
	`0X_CAFE_f00d = 0xcafef00d`,

	// octals
	`0o0 = 0`,
	`0o1234 = 01234`,
	`0o01234567 = 01234567`,

	`0O0 = 0`,
	`0O1234 = 01234`,
	`0O01234567 = 01234567`,

	`0o_0 = 0`,
	`0o_1234 = 01234`,
	`0o0123_4567 = 01234567`,

	`0O_0 = 0`,
	`0O_1234 = 01234`,
	`0O0123_4567 = 01234567`,

	// binaries
	`0b0 = 0`,
	`0b1011 = 0xb`,
	`0b00101101 = 0x2d`,

	`0B0 = 0`,
	`0B1011 = 0xb`,
	`0B00101101 = 0x2d`,

	`0b_0 = 0`,
	`0b10_11 = 0xb`,
	`0b_0010_1101 = 0x2d`,
}

// The RHS operand may be a floating-point quotient n/d of two integer values n and d.
var floatTests = []string{
	// decimal floats
	`1_2_3. = 123.`,
	`0_123. = 123.`,

	`0_0e0 = 0.`,
	`1_2_3e0 = 123.`,
	`0_123e0 = 123.`,

	`0e-0_0 = 0.`,
	`1_2_3E+0 = 123.`,
	`0123E1_2_3 = 123e123`,

	`0.e+1 = 0.`,
	`123.E-1_0 = 123e-10`,
	`01_23.e123 = 123e123`,

	`.0e-1 = .0`,
	`.123E+10 = .123e10`,
	`.0123E123 = .0123e123`,

	`1_2_3.123 = 123.123`,
	`0123.01_23 = 123.0123`,

	`1e-1000000000 = 0`,
	`1e+1000000000 = ?`,
	`6e5518446744 = ?`,
	`-6e5518446744 = ?`,

	// hexadecimal floats
	`0x0.p+0 = 0.`,
	`0Xdeadcafe.p-10 = 0xdeadcafe/1024`,
	`0x1234.P84 = 0x1234000000000000000000000`,

	`0x.1p-0 = 1/16`,
	`0X.deadcafep4 = 0xdeadcafe/0x10000000`,
	`0x.1234P+12 = 0x1234/0x10`,

	`0x0p0 = 0.`,
	`0Xdeadcafep+1 = 0x1bd5b95fc`,
	`0x1234P-10 = 0x1234/1024`,

	`0x0.0p0 = 0.`,
	`0Xdead.cafep+1 = 0x1bd5b95fc/0x10000`,
	`0x12.34P-10 = 0x1234/0x40000`,

	`0Xdead_cafep+1 = 0xdeadcafep+1`,
	`0x_1234P-10 = 0x1234p-10`,

	`0X_dead_cafe.p-10 = 0xdeadcafe.p-10`,
	`0x12_34.P1_2_3 = 0x1234.p123`,
}

var imagTests = []string{
	`1_234i = 1234i`,
	`1_234_567i = 1234567i`,

	`0.i = 0i`,
	`123.i = 123i`,
	`0123.i = 123i`,

	`0.e+1i = 0i`,
	`123.E-1_0i = 123e-10i`,
	`01_23.e123i = 123e123i`,

	`1e-1000000000i = 0i`,
	`1e+1000000000i = ?`,
	`6e5518446744i = ?`,
	`-6e5518446744i = ?`,
}

func testNumbers(t *testing.T, kind token.Token, tests []string) {
	for _, test := range tests {
		a := strings.Split(test, " = ")
		if len(a) != 2 {
			t.Errorf("invalid test case: %s", test)
			continue
		}

		x := MakeFromLiteral(a[0], kind, 0)
		var y Value
		if a[1] == "?" {
			y = MakeUnknown()
		} else {
			if ns, ds, ok := strings.Cut(a[1], "/"); ok && kind == token.FLOAT {
				n := MakeFromLiteral(ns, token.INT, 0)
				d := MakeFromLiteral(ds, token.INT, 0)
				y = BinaryOp(n, token.QUO, d)
			} else {
				y = MakeFromLiteral(a[1], kind, 0)
			}
			if y.Kind() == Unknown {
				panic(fmt.Sprintf("invalid test case: %s %d", test, y.Kind()))
			}
		}

		xk := x.Kind()
		yk := y.Kind()
		if xk != yk {
			t.Errorf("%s: got kind %d != %d", test, xk, yk)
			continue
		}

		if yk == Unknown {
			continue
		}

		if !Compare(x, token.EQL, y) {
			t.Errorf("%s: %s != %s", test, x, y)
		}
	}
}

// TestNumbers verifies that differently written literals
// representing the same number do have the same value.
func TestNumbers(t *testing.T) {
	testNumbers(t, token.INT, intTests)
	testNumbers(t, token.FLOAT, floatTests)
	testNumbers(t, token.IMAG, imagTests)
}

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
	`0 * 1e+1000000000 = ?`,

	`0 / 0 = "division_by_zero"`,
	`10 / 2 = 5`,
	`5 / 3 = 5/3`,
	`5i / 3i = 5/3`,
	`? / 0 = ?`,
	`0 / ? = ?`,
	`0 * 1e+1000000000i = ?`,

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
	{"0e9999999999", "0", "0"},   // issue #16176
	{"-6e-1886451601", "0", "0"}, // issue #20228

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

	if as, bs, ok := strings.Cut(lit, "/"); ok {
		// assume fraction
		a := MakeFromLiteral(as, token.INT, 0)
		b := MakeFromLiteral(bs, token.INT, 0)
		return BinaryOp(a, token.QUO, b)
	}

	tok := token.INT
	switch first, last := lit[0], lit[len(lit)-1]; {
	case first == '"' || first == '`':
		tok = token.STRING
		lit = strings.ReplaceAll(lit, "_", " ")
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
		MakeFromLiteral("''", token.CHAR, 0),
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

func TestMakeFloat64(t *testing.T) {
	var zero float64
	for _, arg := range []float64{
		-math.MaxFloat32,
		-10,
		-0.5,
		-zero,
		zero,
		1,
		10,
		123456789.87654321e-23,
		1e10,
		math.MaxFloat64,
	} {
		val := MakeFloat64(arg)
		if val.Kind() != Float {
			t.Errorf("%v: got kind = %d; want %d", arg, val.Kind(), Float)
		}

		// -0.0 is mapped to 0.0
		got, exact := Float64Val(val)
		if !exact || math.Float64bits(got) != math.Float64bits(arg+0) {
			t.Errorf("%v: got %v (exact = %v)", arg, got, exact)
		}
	}

	// infinity
	for sign := range []int{-1, 1} {
		arg := math.Inf(sign)
		val := MakeFloat64(arg)
		if val.Kind() != Unknown {
			t.Errorf("%v: got kind = %d; want %d", arg, val.Kind(), Unknown)
		}
	}
}

type makeTestCase struct {
	kind      Kind
	arg, want any
}

func dup(k Kind, x any) makeTestCase { return makeTestCase{k, x, x} }

func TestMake(t *testing.T) {
	for _, test := range []makeTestCase{
		{Bool, false, false},
		{String, "hello", "hello"},

		{Int, int64(1), int64(1)},
		{Int, big.NewInt(10), int64(10)},
		{Int, new(big.Int).Lsh(big.NewInt(1), 62), int64(1 << 62)},
		dup(Int, new(big.Int).Lsh(big.NewInt(1), 63)),

		{Float, big.NewFloat(0), floatVal0.val},
		dup(Float, big.NewFloat(2.0)),
		dup(Float, big.NewRat(1, 3)),
	} {
		val := Make(test.arg)
		got := Val(val)
		if val.Kind() != test.kind || got != test.want {
			t.Errorf("got %v (%T, kind = %d); want %v (%T, kind = %d)",
				got, got, val.Kind(), test.want, test.want, test.kind)
		}
	}
}

func BenchmarkStringAdd(b *testing.B) {
	for size := 1; size <= 65536; size *= 4 {
		b.Run(fmt.Sprint(size), func(b *testing.B) {
			b.ReportAllocs()
			n := int64(0)
			for i := 0; i < b.N; i++ {
				x := MakeString(strings.Repeat("x", 100))
				y := x
				for j := 0; j < size-1; j++ {
					y = BinaryOp(y, token.ADD, x)
				}
				n += int64(len(StringVal(y)))
			}
			if n != int64(b.N)*int64(size)*100 {
				b.Fatalf("bad string %d != %d", n, int64(b.N)*int64(size)*100)
			}
		})
	}
}

var bitLenTests = []struct {
	val  int64
	want int
}{
	{0, 0},
	{1, 1},
	{-16, 5},
	{1 << 61, 62},
	{1 << 62, 63},
	{-1 << 62, 63},
	{-1 << 63, 64},
}

func TestBitLen(t *testing.T) {
	for _, test := range bitLenTests {
		if got := BitLen(MakeInt64(test.val)); got != test.want {
			t.Errorf("%v: got %v, want %v", test.val, got, test.want)
		}
	}
}
