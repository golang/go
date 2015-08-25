// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big

import (
	"bytes"
	"fmt"
	"math"
	"strconv"
	"strings"
	"testing"
)

type StringTest struct {
	in, out string
	ok      bool
}

var setStringTests = []StringTest{
	{"0", "0", true},
	{"-0", "0", true},
	{"1", "1", true},
	{"-1", "-1", true},
	{"1.", "1", true},
	{"1e0", "1", true},
	{"1.e1", "10", true},
	{in: "1e"},
	{in: "1.e"},
	{in: "1e+14e-5"},
	{in: "1e4.5"},
	{in: "r"},
	{in: "a/b"},
	{in: "a.b"},
	{"-0.1", "-1/10", true},
	{"-.1", "-1/10", true},
	{"2/4", "1/2", true},
	{".25", "1/4", true},
	{"-1/5", "-1/5", true},
	{"8129567.7690E14", "812956776900000000000", true},
	{"78189e+4", "781890000", true},
	{"553019.8935e+8", "55301989350000", true},
	{"98765432109876543210987654321e-10", "98765432109876543210987654321/10000000000", true},
	{"9877861857500000E-7", "3951144743/4", true},
	{"2169378.417e-3", "2169378417/1000000", true},
	{"884243222337379604041632732738665534", "884243222337379604041632732738665534", true},
	{"53/70893980658822810696", "53/70893980658822810696", true},
	{"106/141787961317645621392", "53/70893980658822810696", true},
	{"204211327800791583.81095", "4084226556015831676219/20000", true},
	{in: "1/0"},
}

// These are not supported by fmt.Fscanf.
var setStringTests2 = []StringTest{
	{"0x10", "16", true},
	{"-010/1", "-8", true}, // TODO(gri) should we even permit octal here?
	{"-010.", "-10", true},
	{"0x10/0x20", "1/2", true},
	{"0b1000/3", "8/3", true},
	// TODO(gri) add more tests
}

func TestRatSetString(t *testing.T) {
	var tests []StringTest
	tests = append(tests, setStringTests...)
	tests = append(tests, setStringTests2...)

	for i, test := range tests {
		x, ok := new(Rat).SetString(test.in)

		if ok {
			if !test.ok {
				t.Errorf("#%d SetString(%q) expected failure", i, test.in)
			} else if x.RatString() != test.out {
				t.Errorf("#%d SetString(%q) got %s want %s", i, test.in, x.RatString(), test.out)
			}
		} else if x != nil {
			t.Errorf("#%d SetString(%q) got %p want nil", i, test.in, x)
		}
	}
}

func TestRatScan(t *testing.T) {
	var buf bytes.Buffer
	for i, test := range setStringTests {
		x := new(Rat)
		buf.Reset()
		buf.WriteString(test.in)

		_, err := fmt.Fscanf(&buf, "%v", x)
		if err == nil != test.ok {
			if test.ok {
				t.Errorf("#%d (%s) error: %s", i, test.in, err)
			} else {
				t.Errorf("#%d (%s) expected error", i, test.in)
			}
			continue
		}
		if err == nil && x.RatString() != test.out {
			t.Errorf("#%d got %s want %s", i, x.RatString(), test.out)
		}
	}
}

var floatStringTests = []struct {
	in   string
	prec int
	out  string
}{
	{"0", 0, "0"},
	{"0", 4, "0.0000"},
	{"1", 0, "1"},
	{"1", 2, "1.00"},
	{"-1", 0, "-1"},
	{"0.05", 1, "0.1"},
	{"-0.05", 1, "-0.1"},
	{".25", 2, "0.25"},
	{".25", 1, "0.3"},
	{".25", 3, "0.250"},
	{"-1/3", 3, "-0.333"},
	{"-2/3", 4, "-0.6667"},
	{"0.96", 1, "1.0"},
	{"0.999", 2, "1.00"},
	{"0.9", 0, "1"},
	{".25", -1, "0"},
	{".55", -1, "1"},
}

func TestFloatString(t *testing.T) {
	for i, test := range floatStringTests {
		x, _ := new(Rat).SetString(test.in)

		if x.FloatString(test.prec) != test.out {
			t.Errorf("#%d got %s want %s", i, x.FloatString(test.prec), test.out)
		}
	}
}

// Test inputs to Rat.SetString.  The prefix "long:" causes the test
// to be skipped in --test.short mode.  (The threshold is about 500us.)
var float64inputs = []string{
	// Constants plundered from strconv/testfp.txt.

	// Table 1: Stress Inputs for Conversion to 53-bit Binary, < 1/2 ULP
	"5e+125",
	"69e+267",
	"999e-026",
	"7861e-034",
	"75569e-254",
	"928609e-261",
	"9210917e+080",
	"84863171e+114",
	"653777767e+273",
	"5232604057e-298",
	"27235667517e-109",
	"653532977297e-123",
	"3142213164987e-294",
	"46202199371337e-072",
	"231010996856685e-073",
	"9324754620109615e+212",
	"78459735791271921e+049",
	"272104041512242479e+200",
	"6802601037806061975e+198",
	"20505426358836677347e-221",
	"836168422905420598437e-234",
	"4891559871276714924261e+222",

	// Table 2: Stress Inputs for Conversion to 53-bit Binary, > 1/2 ULP
	"9e-265",
	"85e-037",
	"623e+100",
	"3571e+263",
	"81661e+153",
	"920657e-023",
	"4603285e-024",
	"87575437e-309",
	"245540327e+122",
	"6138508175e+120",
	"83356057653e+193",
	"619534293513e+124",
	"2335141086879e+218",
	"36167929443327e-159",
	"609610927149051e-255",
	"3743626360493413e-165",
	"94080055902682397e-242",
	"899810892172646163e+283",
	"7120190517612959703e+120",
	"25188282901709339043e-252",
	"308984926168550152811e-052",
	"6372891218502368041059e+064",

	// Table 14: Stress Inputs for Conversion to 24-bit Binary, <1/2 ULP
	"5e-20",
	"67e+14",
	"985e+15",
	"7693e-42",
	"55895e-16",
	"996622e-44",
	"7038531e-32",
	"60419369e-46",
	"702990899e-20",
	"6930161142e-48",
	"25933168707e+13",
	"596428896559e+20",

	// Table 15: Stress Inputs for Conversion to 24-bit Binary, >1/2 ULP
	"3e-23",
	"57e+18",
	"789e-35",
	"2539e-18",
	"76173e+28",
	"887745e-11",
	"5382571e-37",
	"82381273e-35",
	"750486563e-38",
	"3752432815e-39",
	"75224575729e-45",
	"459926601011e+15",

	// Constants plundered from strconv/atof_test.go.

	"0",
	"1",
	"+1",
	"1e23",
	"1E23",
	"100000000000000000000000",
	"1e-100",
	"123456700",
	"99999999999999974834176",
	"100000000000000000000001",
	"100000000000000008388608",
	"100000000000000016777215",
	"100000000000000016777216",
	"-1",
	"-0.1",
	"-0", // NB: exception made for this input
	"1e-20",
	"625e-3",

	// largest float64
	"1.7976931348623157e308",
	"-1.7976931348623157e308",
	// next float64 - too large
	"1.7976931348623159e308",
	"-1.7976931348623159e308",
	// the border is ...158079
	// borderline - okay
	"1.7976931348623158e308",
	"-1.7976931348623158e308",
	// borderline - too large
	"1.797693134862315808e308",
	"-1.797693134862315808e308",

	// a little too large
	"1e308",
	"2e308",
	"1e309",

	// way too large
	"1e310",
	"-1e310",
	"1e400",
	"-1e400",
	"long:1e400000",
	"long:-1e400000",

	// denormalized
	"1e-305",
	"1e-306",
	"1e-307",
	"1e-308",
	"1e-309",
	"1e-310",
	"1e-322",
	// smallest denormal
	"5e-324",
	"4e-324",
	"3e-324",
	// too small
	"2e-324",
	// way too small
	"1e-350",
	"long:1e-400000",
	// way too small, negative
	"-1e-350",
	"long:-1e-400000",

	// try to overflow exponent
	// [Disabled: too slow and memory-hungry with rationals.]
	// "1e-4294967296",
	// "1e+4294967296",
	// "1e-18446744073709551616",
	// "1e+18446744073709551616",

	// http://www.exploringbinary.com/java-hangs-when-converting-2-2250738585072012e-308/
	"2.2250738585072012e-308",
	// http://www.exploringbinary.com/php-hangs-on-numeric-value-2-2250738585072011e-308/
	"2.2250738585072011e-308",

	// A very large number (initially wrongly parsed by the fast algorithm).
	"4.630813248087435e+307",

	// A different kind of very large number.
	"22.222222222222222",
	"long:2." + strings.Repeat("2", 4000) + "e+1",

	// Exactly halfway between 1 and math.Nextafter(1, 2).
	// Round to even (down).
	"1.00000000000000011102230246251565404236316680908203125",
	// Slightly lower; still round down.
	"1.00000000000000011102230246251565404236316680908203124",
	// Slightly higher; round up.
	"1.00000000000000011102230246251565404236316680908203126",
	// Slightly higher, but you have to read all the way to the end.
	"long:1.00000000000000011102230246251565404236316680908203125" + strings.Repeat("0", 10000) + "1",

	// Smallest denormal, 2^(-1022-52)
	"4.940656458412465441765687928682213723651e-324",
	// Half of smallest denormal, 2^(-1022-53)
	"2.470328229206232720882843964341106861825e-324",
	// A little more than the exact half of smallest denormal
	// 2^-1075 + 2^-1100.  (Rounds to 1p-1074.)
	"2.470328302827751011111470718709768633275e-324",
	// The exact halfway between smallest normal and largest denormal:
	// 2^-1022 - 2^-1075.  (Rounds to 2^-1022.)
	"2.225073858507201136057409796709131975935e-308",

	"1152921504606846975",  //   1<<60 - 1
	"-1152921504606846975", // -(1<<60 - 1)
	"1152921504606846977",  //   1<<60 + 1
	"-1152921504606846977", // -(1<<60 + 1)

	"1/3",
}

// isFinite reports whether f represents a finite rational value.
// It is equivalent to !math.IsNan(f) && !math.IsInf(f, 0).
func isFinite(f float64) bool {
	return math.Abs(f) <= math.MaxFloat64
}

func TestFloat32SpecialCases(t *testing.T) {
	for _, input := range float64inputs {
		if strings.HasPrefix(input, "long:") {
			if testing.Short() {
				continue
			}
			input = input[len("long:"):]
		}

		r, ok := new(Rat).SetString(input)
		if !ok {
			t.Errorf("Rat.SetString(%q) failed", input)
			continue
		}
		f, exact := r.Float32()

		// 1. Check string -> Rat -> float32 conversions are
		// consistent with strconv.ParseFloat.
		// Skip this check if the input uses "a/b" rational syntax.
		if !strings.Contains(input, "/") {
			e64, _ := strconv.ParseFloat(input, 32)
			e := float32(e64)

			// Careful: negative Rats too small for
			// float64 become -0, but Rat obviously cannot
			// preserve the sign from SetString("-0").
			switch {
			case math.Float32bits(e) == math.Float32bits(f):
				// Ok: bitwise equal.
			case f == 0 && r.Num().BitLen() == 0:
				// Ok: Rat(0) is equivalent to both +/- float64(0).
			default:
				t.Errorf("strconv.ParseFloat(%q) = %g (%b), want %g (%b); delta = %g", input, e, e, f, f, f-e)
			}
		}

		if !isFinite(float64(f)) {
			continue
		}

		// 2. Check f is best approximation to r.
		if !checkIsBestApprox32(t, f, r) {
			// Append context information.
			t.Errorf("(input was %q)", input)
		}

		// 3. Check f->R->f roundtrip is non-lossy.
		checkNonLossyRoundtrip32(t, f)

		// 4. Check exactness using slow algorithm.
		if wasExact := new(Rat).SetFloat64(float64(f)).Cmp(r) == 0; wasExact != exact {
			t.Errorf("Rat.SetString(%q).Float32().exact = %t, want %t", input, exact, wasExact)
		}
	}
}

func TestFloat64SpecialCases(t *testing.T) {
	for _, input := range float64inputs {
		if strings.HasPrefix(input, "long:") {
			if testing.Short() {
				continue
			}
			input = input[len("long:"):]
		}

		r, ok := new(Rat).SetString(input)
		if !ok {
			t.Errorf("Rat.SetString(%q) failed", input)
			continue
		}
		f, exact := r.Float64()

		// 1. Check string -> Rat -> float64 conversions are
		// consistent with strconv.ParseFloat.
		// Skip this check if the input uses "a/b" rational syntax.
		if !strings.Contains(input, "/") {
			e, _ := strconv.ParseFloat(input, 64)

			// Careful: negative Rats too small for
			// float64 become -0, but Rat obviously cannot
			// preserve the sign from SetString("-0").
			switch {
			case math.Float64bits(e) == math.Float64bits(f):
				// Ok: bitwise equal.
			case f == 0 && r.Num().BitLen() == 0:
				// Ok: Rat(0) is equivalent to both +/- float64(0).
			default:
				t.Errorf("strconv.ParseFloat(%q) = %g (%b), want %g (%b); delta = %g", input, e, e, f, f, f-e)
			}
		}

		if !isFinite(f) {
			continue
		}

		// 2. Check f is best approximation to r.
		if !checkIsBestApprox64(t, f, r) {
			// Append context information.
			t.Errorf("(input was %q)", input)
		}

		// 3. Check f->R->f roundtrip is non-lossy.
		checkNonLossyRoundtrip64(t, f)

		// 4. Check exactness using slow algorithm.
		if wasExact := new(Rat).SetFloat64(f).Cmp(r) == 0; wasExact != exact {
			t.Errorf("Rat.SetString(%q).Float64().exact = %t, want %t", input, exact, wasExact)
		}
	}
}
