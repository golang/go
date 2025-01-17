// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big

import (
	"fmt"
	"math"
	"math/bits"
	"math/rand/v2"
	"runtime"
	"strings"
	"testing"
)

var cmpTests = []struct {
	x, y nat
	r    int
}{
	{nil, nil, 0},
	{nil, nat(nil), 0},
	{nat(nil), nil, 0},
	{nat(nil), nat(nil), 0},
	{nat{0}, nat{0}, 0},
	{nat{0}, nat{1}, -1},
	{nat{1}, nat{0}, 1},
	{nat{1}, nat{1}, 0},
	{nat{0, _M}, nat{1}, 1},
	{nat{1}, nat{0, _M}, -1},
	{nat{1, _M}, nat{0, _M}, 1},
	{nat{0, _M}, nat{1, _M}, -1},
	{nat{16, 571956, 8794, 68}, nat{837, 9146, 1, 754489}, -1},
	{nat{34986, 41, 105, 1957}, nat{56, 7458, 104, 1957}, 1},
}

func TestCmp(t *testing.T) {
	for i, a := range cmpTests {
		r := a.x.cmp(a.y)
		if r != a.r {
			t.Errorf("#%d got r = %v; want %v", i, r, a.r)
		}
	}
}

type funNN func(z, x, y nat) nat
type funSNN func(z nat, stk *stack, x, y nat) nat
type argNN struct {
	z, x, y nat
}

var sumNN = []argNN{
	{},
	{nat{1}, nil, nat{1}},
	{nat{1111111110}, nat{123456789}, nat{987654321}},
	{nat{0, 0, 0, 1}, nil, nat{0, 0, 0, 1}},
	{nat{0, 0, 0, 1111111110}, nat{0, 0, 0, 123456789}, nat{0, 0, 0, 987654321}},
	{nat{0, 0, 0, 1}, nat{0, 0, _M}, nat{0, 0, 1}},
}

var prodNN = append(prodTests(), prodNNExtra...)

func permute[E any](x []E) {
	out := make([]E, len(x))
	for i, j := range rand.Perm(len(x)) {
		out[i] = x[j]
	}
	copy(x, out)
}

// testMul returns the product of x and y using the grade-school algorithm,
// as a reference implementation.
func testMul(x, y nat) nat {
	z := make(nat, len(x)+len(y))
	for i, xi := range x {
		for j, yj := range y {
			hi, lo := bits.Mul(uint(xi), uint(yj))
			k := i + j
			s, c := bits.Add(uint(z[k]), lo, 0)
			z[k] = Word(s)
			k++
			for hi != 0 || c != 0 {
				s, c = bits.Add(uint(z[k]), hi, c)
				hi = 0
				z[k] = Word(s)
				k++
			}
		}
	}
	return z.norm()
}

func prodTests() []argNN {
	var tests []argNN
	for size := range 10 {
		var x, y nat
		for i := range size {
			x = append(x, Word(i+1))
			y = append(y, Word(i+1+size))
		}
		permute(x)
		permute(y)
		x = x.norm()
		y = y.norm()
		tests = append(tests, argNN{testMul(x, y), x, y})
	}

	words := []Word{0, 1, 2, 3, 4, ^Word(0), ^Word(1), ^Word(2), ^Word(3)}
	for size := range 10 {
		if size == 0 {
			continue // already tested the only 0-length possibility above
		}
		for range 10 {
			x := make(nat, size)
			y := make(nat, size)
			for i := range size {
				x[i] = words[rand.N(len(words))]
				y[i] = words[rand.N(len(words))]
			}
			x = x.norm()
			y = y.norm()
			tests = append(tests, argNN{testMul(x, y), x, y})
		}
	}
	return tests
}

var prodNNExtra = []argNN{
	{nil, nat{991}, nil},
	{nat{991}, nat{991}, nat{1}},
	{nat{991 * 991}, nat{991}, nat{991}},
	{nat{8, 22, 15}, nat{2, 3}, nat{4, 5}},
	{nat{10, 27, 52, 45, 28}, nat{2, 3, 4}, nat{5, 6, 7}},
	{nat{12, 32, 61, 100, 94, 76, 45}, nat{2, 3, 4, 5}, nat{6, 7, 8, 9}},
	{nat{12, 32, 61, 100, 94, 76, 45}, nat{2, 3, 4, 5}, nat{6, 7, 8, 9}},
	{nat{14, 37, 70, 114, 170, 166, 148, 115, 66}, nat{2, 3, 4, 5, 6}, nat{7, 8, 9, 10, 11}},
	{nat{991 * 991, 991 * 2, 1}, nat{991, 1}, nat{991, 1}},
	{nat{991 * 991, 991 * 777 * 2, 777 * 777}, nat{991, 777}, nat{991, 777}},
	{nat{0, 0, 991 * 991}, nat{0, 991}, nat{0, 991}},
	{nat{1 * 991, 2 * 991, 3 * 991, 4 * 991}, nat{1, 2, 3, 4}, nat{991}},
	{nat{4, 11, 20, 30, 20, 11, 4}, nat{1, 2, 3, 4}, nat{4, 3, 2, 1}},
	// 3^100 * 3^28 = 3^128
	{
		natFromString("11790184577738583171520872861412518665678211592275841109096961"),
		natFromString("515377520732011331036461129765621272702107522001"),
		natFromString("22876792454961"),
	},
	// z = 111....1 (70000 digits)
	// x = 10^(99*700) + ... + 10^1400 + 10^700 + 1
	// y = 111....1 (700 digits, larger than Karatsuba threshold on 32-bit and 64-bit)
	{
		natFromString(strings.Repeat("1", 70000)),
		natFromString("1" + strings.Repeat(strings.Repeat("0", 699)+"1", 99)),
		natFromString(strings.Repeat("1", 700)),
	},
	// z = 111....1 (20000 digits)
	// x = 10^10000 + 1
	// y = 111....1 (10000 digits)
	{
		natFromString(strings.Repeat("1", 20000)),
		natFromString("1" + strings.Repeat("0", 9999) + "1"),
		natFromString(strings.Repeat("1", 10000)),
	},
}

func natFromString(s string) nat {
	x, _, _, err := nat(nil).scan(strings.NewReader(s), 0, false)
	if err != nil {
		panic(err)
	}
	return x
}

func TestSet(t *testing.T) {
	for _, a := range sumNN {
		z := nat(nil).set(a.z)
		if z.cmp(a.z) != 0 {
			t.Errorf("got z = %v; want %v", z, a.z)
		}
	}
}

func testFunNN(t *testing.T, msg string, f funNN, a argNN) {
	z := f(nil, a.x, a.y)
	if z.cmp(a.z) != 0 {
		t.Errorf("%s%+v\n\tgot z = %v; want %v", msg, a, z, a.z)
	}
}

func testFunSNN(t *testing.T, msg string, f funSNN, a argNN) {
	t.Helper()
	stk := getStack()
	defer stk.free()
	z := f(nil, stk, a.x, a.y)
	if z.cmp(a.z) != 0 {
		t.Fatalf("%s%+v\n\tgot z = %v; want %v", msg, a, z, a.z)
	}
}

func setDuringTest[V any](t *testing.T, p *V, v V) {
	old := *p
	*p = v
	t.Cleanup(func() { *p = old })
}

func TestAdd(t *testing.T) {
	for _, a := range sumNN {
		testFunNN(t, "add", nat.add, a)
		a.x, a.y = a.y, a.x
		testFunNN(t, "add", nat.add, a)
	}
}

func TestSub(t *testing.T) {
	for _, a := range sumNN {
		a.x, a.z = a.z, a.x
		testFunNN(t, "sub", nat.sub, a)

		a.y, a.z = a.z, a.y
		testFunNN(t, "sub", nat.sub, a)
	}
}

func TestNatMul(t *testing.T) {
	t.Run("Basic", func(t *testing.T) {
		setDuringTest(t, &karatsubaThreshold, 1e9)
		for _, a := range prodNN {
			if len(a.z) >= 100 {
				continue
			}
			testFunSNN(t, "mul", nat.mul, a)
			a.x, a.y = a.y, a.x
			testFunSNN(t, "mul", nat.mul, a)
		}
	})
	t.Run("Karatsuba", func(t *testing.T) {
		setDuringTest(t, &karatsubaThreshold, 2)
		for _, a := range prodNN {
			testFunSNN(t, "mul", nat.mul, a)
			a.x, a.y = a.y, a.x
			testFunSNN(t, "mul", nat.mul, a)
		}
	})

	t.Run("Mul", func(t *testing.T) {
		for _, a := range prodNN {
			testFunSNN(t, "mul", nat.mul, a)
			a.x, a.y = a.y, a.x
			testFunSNN(t, "mul", nat.mul, a)
		}
	})
}

func testSqr(t *testing.T, x nat) {
	stk := getStack()
	defer stk.free()

	got := make(nat, 2*len(x))
	want := make(nat, 2*len(x))
	got = got.sqr(stk, x)
	want = want.mul(stk, x, x)
	if got.cmp(want) != 0 {
		t.Errorf("basicSqr(%v), got %v, want %v", x, got, want)
	}
}

func TestNatSqr(t *testing.T) {
	t.Run("Basic", func(t *testing.T) {
		setDuringTest(t, &basicSqrThreshold, 0)
		setDuringTest(t, &karatsubaSqrThreshold, 1e9)
		for _, a := range prodNN {
			if len(a.z) >= 100 {
				continue
			}
			testSqr(t, a.x)
			testSqr(t, a.y)
			testSqr(t, a.z)
		}
	})
	t.Run("Karatsuba", func(t *testing.T) {
		setDuringTest(t, &basicSqrThreshold, 2)
		setDuringTest(t, &karatsubaSqrThreshold, 2)
		for _, a := range prodNN {
			testSqr(t, a.x)
			testSqr(t, a.y)
			testSqr(t, a.z)
		}
	})
	t.Run("Sqr", func(t *testing.T) {
		for _, a := range prodNN {
			testSqr(t, a.x)
			testSqr(t, a.y)
			testSqr(t, a.z)
		}
	})
}

var mulRangesN = []struct {
	a, b uint64
	prod string
}{
	{0, 0, "0"},
	{1, 1, "1"},
	{1, 2, "2"},
	{1, 3, "6"},
	{10, 10, "10"},
	{0, 100, "0"},
	{0, 1e9, "0"},
	{1, 0, "1"},                    // empty range
	{100, 1, "1"},                  // empty range
	{1, 10, "3628800"},             // 10!
	{1, 20, "2432902008176640000"}, // 20!
	{1, 100,
		"933262154439441526816992388562667004907159682643816214685929" +
			"638952175999932299156089414639761565182862536979208272237582" +
			"51185210916864000000000000000000000000", // 100!
	},
	{math.MaxUint64 - 0, math.MaxUint64, "18446744073709551615"},
	{math.MaxUint64 - 1, math.MaxUint64, "340282366920938463408034375210639556610"},
	{math.MaxUint64 - 2, math.MaxUint64, "6277101735386680761794095221682035635525021984684230311930"},
	{math.MaxUint64 - 3, math.MaxUint64, "115792089237316195360799967654821100226821973275796746098729803619699194331160"},
}

func TestMulRangeN(t *testing.T) {
	stk := getStack()
	defer stk.free()

	for i, r := range mulRangesN {
		prod := string(nat(nil).mulRange(stk, r.a, r.b).utoa(10))
		if prod != r.prod {
			t.Errorf("#%d: got %s; want %s", i, prod, r.prod)
		}
	}
}

// allocBytes returns the number of bytes allocated by invoking f.
func allocBytes(f func()) uint64 {
	var stats runtime.MemStats
	runtime.ReadMemStats(&stats)
	t := stats.TotalAlloc
	f()
	runtime.ReadMemStats(&stats)
	return stats.TotalAlloc - t
}

// TestMulUnbalanced tests that multiplying numbers of different lengths
// does not cause deep recursion and in turn allocate too much memory.
// Test case for issue 3807.
func TestMulUnbalanced(t *testing.T) {
	stk := getStack()
	defer stk.free()

	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))
	x := rndNat(50000)
	y := rndNat(40)
	allocSize := allocBytes(func() {
		nat(nil).mul(stk, x, y)
	})
	inputSize := uint64(len(x)+len(y)) * _S
	if ratio := allocSize / uint64(inputSize); ratio > 10 {
		t.Errorf("multiplication uses too much memory (%d > %d times the size of inputs)", allocSize, ratio)
	}
}

// rndNat returns a random nat value >= 0 of (usually) n words in length.
// In extremely unlikely cases it may be smaller than n words if the top-
// most words are 0.
func rndNat(n int) nat {
	return nat(rndV(n)).norm()
}

// rndNat1 is like rndNat but the result is guaranteed to be > 0.
func rndNat1(n int) nat {
	x := nat(rndV(n)).norm()
	if len(x) == 0 {
		x.setWord(1)
	}
	return x
}

func BenchmarkMul(b *testing.B) {
	stk := getStack()
	defer stk.free()

	mulx := rndNat(1e4)
	muly := rndNat(1e4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var z nat
		z.mul(stk, mulx, muly)
	}
}

func benchmarkNatMul(b *testing.B, nwords int) {
	x := rndNat(nwords)
	y := rndNat(nwords)
	var z nat
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		z.mul(nil, x, y)
	}
}

var mulBenchSizes = []int{10, 100, 1000, 10000, 100000}

func BenchmarkNatMul(b *testing.B) {
	for _, n := range mulBenchSizes {
		if isRaceBuilder && n > 1e3 {
			continue
		}
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			benchmarkNatMul(b, n)
		})
	}
}

func TestNLZ(t *testing.T) {
	var x Word = _B >> 1
	for i := 0; i <= _W; i++ {
		if int(nlz(x)) != i {
			t.Errorf("failed at %x: got %d want %d", x, nlz(x), i)
		}
		x >>= 1
	}
}

type shiftTest struct {
	in    nat
	shift uint
	out   nat
}

var leftShiftTests = []shiftTest{
	{nil, 0, nil},
	{nil, 1, nil},
	{natOne, 0, natOne},
	{natOne, 1, natTwo},
	{nat{1 << (_W - 1)}, 1, nat{0}},
	{nat{1 << (_W - 1), 0}, 1, nat{0, 1}},
}

func TestShiftLeft(t *testing.T) {
	for i, test := range leftShiftTests {
		var z nat
		z = z.shl(test.in, test.shift)
		for j, d := range test.out {
			if j >= len(z) || z[j] != d {
				t.Errorf("#%d: got: %v want: %v", i, z, test.out)
				break
			}
		}
	}
}

var rightShiftTests = []shiftTest{
	{nil, 0, nil},
	{nil, 1, nil},
	{natOne, 0, natOne},
	{natOne, 1, nil},
	{natTwo, 1, natOne},
	{nat{0, 1}, 1, nat{1 << (_W - 1)}},
	{nat{2, 1, 1}, 1, nat{1<<(_W-1) + 1, 1 << (_W - 1)}},
}

func TestShiftRight(t *testing.T) {
	for i, test := range rightShiftTests {
		var z nat
		z = z.shr(test.in, test.shift)
		for j, d := range test.out {
			if j >= len(z) || z[j] != d {
				t.Errorf("#%d: got: %v want: %v", i, z, test.out)
				break
			}
		}
	}
}

func BenchmarkZeroShifts(b *testing.B) {
	x := rndNat(800)

	b.Run("Shl", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			var z nat
			z.shl(x, 0)
		}
	})
	b.Run("ShlSame", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			x.shl(x, 0)
		}
	})

	b.Run("Shr", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			var z nat
			z.shr(x, 0)
		}
	})
	b.Run("ShrSame", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			x.shr(x, 0)
		}
	})
}

type modWTest struct {
	in       string
	dividend string
	out      string
}

var modWTests32 = []modWTest{
	{"23492635982634928349238759823742", "252341", "220170"},
}

var modWTests64 = []modWTest{
	{"6527895462947293856291561095690465243862946", "524326975699234", "375066989628668"},
}

func runModWTests(t *testing.T, tests []modWTest) {
	for i, test := range tests {
		in, _ := new(Int).SetString(test.in, 10)
		d, _ := new(Int).SetString(test.dividend, 10)
		out, _ := new(Int).SetString(test.out, 10)

		r := in.abs.modW(d.abs[0])
		if r != out.abs[0] {
			t.Errorf("#%d failed: got %d want %s", i, r, out)
		}
	}
}

func TestModW(t *testing.T) {
	if _W >= 32 {
		runModWTests(t, modWTests32)
	}
	if _W >= 64 {
		runModWTests(t, modWTests64)
	}
}

var montgomeryTests = []struct {
	x, y, m      string
	k0           uint64
	out32, out64 string
}{
	{
		"0xffffffffffffffffffffffffffffffffffffffffffffffffe",
		"0xffffffffffffffffffffffffffffffffffffffffffffffffe",
		"0xfffffffffffffffffffffffffffffffffffffffffffffffff",
		1,
		"0x1000000000000000000000000000000000000000000",
		"0x10000000000000000000000000000000000",
	},
	{
		"0x000000000ffffff5",
		"0x000000000ffffff0",
		"0x0000000010000001",
		0xff0000000fffffff,
		"0x000000000bfffff4",
		"0x0000000003400001",
	},
	{
		"0x0000000080000000",
		"0x00000000ffffffff",
		"0x1000000000000001",
		0xfffffffffffffff,
		"0x0800000008000001",
		"0x0800000008000001",
	},
	{
		"0x0000000080000000",
		"0x0000000080000000",
		"0xffffffff00000001",
		0xfffffffeffffffff,
		"0xbfffffff40000001",
		"0xbfffffff40000001",
	},
	{
		"0x0000000080000000",
		"0x0000000080000000",
		"0x00ffffff00000001",
		0xfffffeffffffff,
		"0xbfffff40000001",
		"0xbfffff40000001",
	},
	{
		"0x0000000080000000",
		"0x0000000080000000",
		"0x0000ffff00000001",
		0xfffeffffffff,
		"0xbfff40000001",
		"0xbfff40000001",
	},
	{
		"0x3321ffffffffffffffffffffffffffff00000000000022222623333333332bbbb888c0",
		"0x3321ffffffffffffffffffffffffffff00000000000022222623333333332bbbb888c0",
		"0x33377fffffffffffffffffffffffffffffffffffffffffffff0000000000022222eee1",
		0xdecc8f1249812adf,
		"0x04eb0e11d72329dc0915f86784820fc403275bf2f6620a20e0dd344c5cd0875e50deb5",
		"0x0d7144739a7d8e11d72329dc0915f86784820fc403275bf2f61ed96f35dd34dbb3d6a0",
	},
	{
		"0x10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000ffffffffffffffffffffffffffffffff00000000000022222223333333333444444444",
		"0x10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000ffffffffffffffffffffffffffffffff999999999999999aaabbbbbbbbcccccccccccc",
		"0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff33377fffffffffffffffffffffffffffffffffffffffffffff0000000000022222eee1",
		0xdecc8f1249812adf,
		"0x5c0d52f451aec609b15da8e5e5626c4eaa88723bdeac9d25ca9b961269400410ca208a16af9c2fb07d7a11c7772cba02c22f9711078d51a3797eb18e691295293284d988e349fa6deba46b25a4ecd9f715",
		"0x92fcad4b5c0d52f451aec609b15da8e5e5626c4eaa88723bdeac9d25ca9b961269400410ca208a16af9c2fb07d799c32fe2f3cc5422f9711078d51a3797eb18e691295293284d8f5e69caf6decddfe1df6",
	},
}

func TestMontgomery(t *testing.T) {
	stk := getStack()
	defer stk.free()

	one := NewInt(1)
	_B := new(Int).Lsh(one, _W)
	for i, test := range montgomeryTests {
		x := natFromString(test.x)
		y := natFromString(test.y)
		m := natFromString(test.m)
		for len(x) < len(m) {
			x = append(x, 0)
		}
		for len(y) < len(m) {
			y = append(y, 0)
		}

		if x.cmp(m) > 0 {
			_, r := nat(nil).div(stk, nil, x, m)
			t.Errorf("#%d: x > m (0x%s > 0x%s; use 0x%s)", i, x.utoa(16), m.utoa(16), r.utoa(16))
		}
		if y.cmp(m) > 0 {
			_, r := nat(nil).div(stk, nil, x, m)
			t.Errorf("#%d: y > m (0x%s > 0x%s; use 0x%s)", i, y.utoa(16), m.utoa(16), r.utoa(16))
		}

		var out nat
		if _W == 32 {
			out = natFromString(test.out32)
		} else {
			out = natFromString(test.out64)
		}

		// t.Logf("#%d: len=%d\n", i, len(m))

		// check output in table
		xi := &Int{abs: x}
		yi := &Int{abs: y}
		mi := &Int{abs: m}
		p := new(Int).Mod(new(Int).Mul(xi, new(Int).Mul(yi, new(Int).ModInverse(new(Int).Lsh(one, uint(len(m))*_W), mi))), mi)
		if out.cmp(p.abs.norm()) != 0 {
			t.Errorf("#%d: out in table=0x%s, computed=0x%s", i, out.utoa(16), p.abs.norm().utoa(16))
		}

		// check k0 in table
		k := new(Int).Mod(&Int{abs: m}, _B)
		k = new(Int).Sub(_B, k)
		k = new(Int).Mod(k, _B)
		k0 := Word(new(Int).ModInverse(k, _B).Uint64())
		if k0 != Word(test.k0) {
			t.Errorf("#%d: k0 in table=%#x, computed=%#x\n", i, test.k0, k0)
		}

		// check montgomery with correct k0 produces correct output
		z := nat(nil).montgomery(x, y, m, k0, len(m))
		z = z.norm()
		if z.cmp(out) != 0 {
			t.Errorf("#%d: got 0x%s want 0x%s", i, z.utoa(16), out.utoa(16))
		}
	}
}

var expNNTests = []struct {
	x, y, m string
	out     string
}{
	{"0", "0", "0", "1"},
	{"0", "0", "1", "0"},
	{"1", "1", "1", "0"},
	{"2", "1", "1", "0"},
	{"2", "2", "1", "0"},
	{"10", "100000000000", "1", "0"},
	{"0x8000000000000000", "2", "", "0x40000000000000000000000000000000"},
	{"0x8000000000000000", "2", "6719", "4944"},
	{"0x8000000000000000", "3", "6719", "5447"},
	{"0x8000000000000000", "1000", "6719", "1603"},
	{"0x8000000000000000", "1000000", "6719", "3199"},
	{
		"2938462938472983472983659726349017249287491026512746239764525612965293865296239471239874193284792387498274256129746192347",
		"298472983472983471903246121093472394872319615612417471234712061",
		"29834729834729834729347290846729561262544958723956495615629569234729836259263598127342374289365912465901365498236492183464",
		"23537740700184054162508175125554701713153216681790245129157191391322321508055833908509185839069455749219131480588829346291",
	},
	{
		"11521922904531591643048817447554701904414021819823889996244743037378330903763518501116638828335352811871131385129455853417360623007349090150042001944696604737499160174391019030572483602867266711107136838523916077674888297896995042968746762200926853379",
		"426343618817810911523",
		"444747819283133684179",
		"42",
	},
	{"375", "249", "388", "175"},
	{"375", "18446744073709551801", "388", "175"},
	{"0", "0x40000000000000", "0x200", "0"},
	{"0xeffffff900002f00", "0x40000000000000", "0x200", "0"},
	{"5", "1435700818", "72", "49"},
	{"0xffff", "0x300030003000300030003000300030003000302a3000300030003000300030003000300030003000300030003000300030003030623066307f3030783062303430383064303630343036", "0x300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", "0xa3f94c08b0b90e87af637cacc9383f7ea032352b8961fc036a52b659b6c9b33491b335ffd74c927f64ddd62cfca0001"},
}

func TestExpNN(t *testing.T) {
	stk := getStack()
	defer stk.free()

	for i, test := range expNNTests {
		x := natFromString(test.x)
		y := natFromString(test.y)
		out := natFromString(test.out)

		var m nat
		if len(test.m) > 0 {
			m = natFromString(test.m)
		}

		z := nat(nil).expNN(stk, x, y, m, false)
		if z.cmp(out) != 0 {
			t.Errorf("#%d got %s want %s", i, z.utoa(10), out.utoa(10))
		}
	}
}

func FuzzExpMont(f *testing.F) {
	f.Fuzz(func(t *testing.T, x1, x2, x3, y1, y2, y3, m1, m2, m3 uint) {
		if m1 == 0 && m2 == 0 && m3 == 0 {
			return
		}
		x := new(Int).SetBits([]Word{Word(x1), Word(x2), Word(x3)})
		y := new(Int).SetBits([]Word{Word(y1), Word(y2), Word(y3)})
		m := new(Int).SetBits([]Word{Word(m1), Word(m2), Word(m3)})
		out := new(Int).Exp(x, y, m)
		want := new(Int).expSlow(x, y, m)
		if out.Cmp(want) != 0 {
			t.Errorf("x = %#x\ny=%#x\nz=%#x\nout=%#x\nwant=%#x\ndc: 16o 16i %X %X %X |p", x, y, m, out, want, x, y, m)
		}
	})
}

func BenchmarkExp3Power(b *testing.B) {
	stk := getStack()
	defer stk.free()

	const x = 3
	for _, y := range []Word{
		0x10, 0x40, 0x100, 0x400, 0x1000, 0x4000, 0x10000, 0x40000, 0x100000, 0x400000,
	} {
		b.Run(fmt.Sprintf("%#x", y), func(b *testing.B) {
			var z nat
			for i := 0; i < b.N; i++ {
				z.expWW(stk, x, y)
			}
		})
	}
}

func fibo(n int) nat {
	switch n {
	case 0:
		return nil
	case 1:
		return nat{1}
	}
	f0 := fibo(0)
	f1 := fibo(1)
	var f2 nat
	for i := 1; i < n; i++ {
		f2 = f2.add(f0, f1)
		f0, f1, f2 = f1, f2, f0
	}
	return f1
}

var fiboNums = []string{
	"0",
	"55",
	"6765",
	"832040",
	"102334155",
	"12586269025",
	"1548008755920",
	"190392490709135",
	"23416728348467685",
	"2880067194370816120",
	"354224848179261915075",
}

func TestFibo(t *testing.T) {
	for i, want := range fiboNums {
		n := i * 10
		got := string(fibo(n).utoa(10))
		if got != want {
			t.Errorf("fibo(%d) failed: got %s want %s", n, got, want)
		}
	}
}

func BenchmarkFibo(b *testing.B) {
	for i := 0; i < b.N; i++ {
		fibo(1e0)
		fibo(1e1)
		fibo(1e2)
		fibo(1e3)
		fibo(1e4)
		fibo(1e5)
	}
}

var bitTests = []struct {
	x    string
	i    uint
	want uint
}{
	{"0", 0, 0},
	{"0", 1, 0},
	{"0", 1000, 0},

	{"0x1", 0, 1},
	{"0x10", 0, 0},
	{"0x10", 3, 0},
	{"0x10", 4, 1},
	{"0x10", 5, 0},

	{"0x8000000000000000", 62, 0},
	{"0x8000000000000000", 63, 1},
	{"0x8000000000000000", 64, 0},

	{"0x3" + strings.Repeat("0", 32), 127, 0},
	{"0x3" + strings.Repeat("0", 32), 128, 1},
	{"0x3" + strings.Repeat("0", 32), 129, 1},
	{"0x3" + strings.Repeat("0", 32), 130, 0},
}

func TestBit(t *testing.T) {
	for i, test := range bitTests {
		x := natFromString(test.x)
		if got := x.bit(test.i); got != test.want {
			t.Errorf("#%d: %s.bit(%d) = %v; want %v", i, test.x, test.i, got, test.want)
		}
	}
}

var stickyTests = []struct {
	x    string
	i    uint
	want uint
}{
	{"0", 0, 0},
	{"0", 1, 0},
	{"0", 1000, 0},

	{"0x1", 0, 0},
	{"0x1", 1, 1},

	{"0x1350", 0, 0},
	{"0x1350", 4, 0},
	{"0x1350", 5, 1},

	{"0x8000000000000000", 63, 0},
	{"0x8000000000000000", 64, 1},

	{"0x1" + strings.Repeat("0", 100), 400, 0},
	{"0x1" + strings.Repeat("0", 100), 401, 1},
}

func TestSticky(t *testing.T) {
	for i, test := range stickyTests {
		x := natFromString(test.x)
		if got := x.sticky(test.i); got != test.want {
			t.Errorf("#%d: %s.sticky(%d) = %v; want %v", i, test.x, test.i, got, test.want)
		}
		if test.want == 1 {
			// all subsequent i's should also return 1
			for d := uint(1); d <= 3; d++ {
				if got := x.sticky(test.i + d); got != 1 {
					t.Errorf("#%d: %s.sticky(%d) = %v; want %v", i, test.x, test.i+d, got, 1)
				}
			}
		}
	}
}

func benchmarkNatSqr(b *testing.B, nwords int) {
	x := rndNat(nwords)
	var z nat
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		z.sqr(nil, x)
	}
}

var sqrBenchSizes = []int{
	1, 2, 3, 5, 8, 10, 20, 30, 50, 80,
	100, 200, 300, 500, 800,
	1000, 10000, 100000,
}

func BenchmarkNatSqr(b *testing.B) {
	for _, n := range sqrBenchSizes {
		if isRaceBuilder && n > 1e3 {
			continue
		}
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			benchmarkNatSqr(b, n)
		})
	}
}

var subMod2NTests = []struct {
	x string
	y string
	n uint
	z string
}{
	{"1", "2", 0, "0"},
	{"1", "0", 1, "1"},
	{"0", "1", 1, "1"},
	{"3", "5", 3, "6"},
	{"5", "3", 3, "2"},
	// 2^65, 2^66-1, 2^65 - (2^66-1) + 2^67
	{"36893488147419103232", "73786976294838206463", 67, "110680464442257309697"},
	// 2^66-1, 2^65, 2^65-1
	{"73786976294838206463", "36893488147419103232", 67, "36893488147419103231"},
}

func TestNatSubMod2N(t *testing.T) {
	for _, mode := range []string{"noalias", "aliasX", "aliasY"} {
		t.Run(mode, func(t *testing.T) {
			for _, tt := range subMod2NTests {
				x0 := natFromString(tt.x)
				y0 := natFromString(tt.y)
				want := natFromString(tt.z)
				x := nat(nil).set(x0)
				y := nat(nil).set(y0)
				var z nat
				switch mode {
				case "aliasX":
					z = x
				case "aliasY":
					z = y
				}
				z = z.subMod2N(x, y, tt.n)
				if z.cmp(want) != 0 {
					t.Fatalf("subMod2N(%d, %d, %d) = %d, want %d", x0, y0, tt.n, z, want)
				}
				if mode != "aliasX" && x.cmp(x0) != 0 {
					t.Fatalf("subMod2N(%d, %d, %d) modified x", x0, y0, tt.n)
				}
				if mode != "aliasY" && y.cmp(y0) != 0 {
					t.Fatalf("subMod2N(%d, %d, %d) modified y", x0, y0, tt.n)
				}
			}
		})
	}
}

func BenchmarkNatSetBytes(b *testing.B) {
	const maxLength = 128
	lengths := []int{
		// No remainder:
		8, 24, maxLength,
		// With remainder:
		7, 23, maxLength - 1,
	}
	n := make(nat, maxLength/_W) // ensure n doesn't need to grow during the test
	buf := make([]byte, maxLength)
	for _, l := range lengths {
		b.Run(fmt.Sprint(l), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				n.setBytes(buf[:l])
			}
		})
	}
}

func TestNatDiv(t *testing.T) {
	stk := getStack()
	defer stk.free()

	sizes := []int{
		1, 2, 5, 8, 15, 25, 40, 65, 100,
		200, 500, 800, 1500, 2500, 4000, 6500, 10000,
	}
	for _, i := range sizes {
		for _, j := range sizes {
			a := rndNat1(i)
			b := rndNat1(j)
			// the test requires b >= 2
			if len(b) == 1 && b[0] == 1 {
				b[0] = 2
			}
			// choose a remainder c < b
			c := rndNat1(len(b))
			if len(c) == len(b) && c[len(c)-1] >= b[len(b)-1] {
				c[len(c)-1] = 0
				c = c.norm()
			}
			// compute x = a*b+c
			x := nat(nil).mul(stk, a, b)
			x = x.add(x, c)

			var q, r nat
			q, r = q.div(stk, r, x, b)
			if q.cmp(a) != 0 {
				t.Fatalf("wrong quotient: got %s; want %s for %s/%s", q.utoa(10), a.utoa(10), x.utoa(10), b.utoa(10))
			}
			if r.cmp(c) != 0 {
				t.Fatalf("wrong remainder: got %s; want %s for %s/%s", r.utoa(10), c.utoa(10), x.utoa(10), b.utoa(10))
			}
		}
	}
}

// TestIssue37499 triggers the edge case of divBasic where
// the inaccurate estimate of the first word's quotient
// happens at the very beginning of the loop.
func TestIssue37499(t *testing.T) {
	stk := getStack()
	defer stk.free()

	// Choose u and v such that v is slightly larger than u >> N.
	// This tricks divBasic into choosing 1 as the first word
	// of the quotient. This works in both 32-bit and 64-bit settings.
	u := natFromString("0x2b6c385a05be027f5c22005b63c42a1165b79ff510e1706b39f8489c1d28e57bb5ba4ef9fd9387a3e344402c0a453381")
	v := natFromString("0x2b6c385a05be027f5c22005b63c42a1165b79ff510e1706c")

	q := nat(nil).make(8)
	q.divBasic(stk, u, v)
	q = q.norm()
	if s := string(q.utoa(16)); s != "fffffffffffffffffffffffffffffffffffffffffffffffb" {
		t.Fatalf("incorrect quotient: %s", s)
	}
}

// TestIssue42552 triggers an edge case of recursive division
// where the first division loop is never entered, and correcting
// the remainder takes exactly two iterations in the final loop.
func TestIssue42552(t *testing.T) {
	stk := getStack()
	defer stk.free()

	u := natFromString("0xc23b166884c3869092a520eceedeced2b00847bd256c9cf3b2c5e2227c15bd5e6ee7ef8a2f49236ad0eedf2c8a3b453cf6e0706f64285c526b372c4b1321245519d430540804a50b7ca8b6f1b34a2ec05cdbc24de7599af112d3e3c8db347e8799fe70f16e43c6566ba3aeb169463a3ecc486172deb2d9b80a3699c776e44fef20036bd946f1b4d054dd88a2c1aeb986199b0b2b7e58c42288824b74934d112fe1fc06e06b4d99fe1c5e725946b23210521e209cd507cce90b5f39a523f27e861f9e232aee50c3f585208b4573dcc0b897b6177f2ba20254fd5c50a033e849dee1b3a93bd2dc44ba8ca836cab2c2ae50e50b126284524fa0187af28628ff0face68d87709200329db1392852c8b8963fbe3d05fb1efe19f0ed5ca9fadc2f96f82187c24bb2512b2e85a66333a7e176605695211e1c8e0b9b9e82813e50654964945b1e1e66a90840396c7d10e23e47f364d2d3f660fa54598e18d1ca2ea4fe4f35a40a11f69f201c80b48eaee3e2e9b0eda63decf92bec08a70f731587d4ed0f218d5929285c8b2ccbc497e20db42de73885191fa453350335990184d8df805072f958d5354debda38f5421effaaafd6cb9b721ace74be0892d77679f62a4a126697cd35797f6858193da4ba1770c06aea2e5c59ec04b8ea26749e61b72ecdde403f3bc7e5e546cd799578cc939fa676dfd5e648576d4a06cbadb028adc2c0b461f145b2321f42e5e0f3b4fb898ecd461df07a6f5154067787bf74b5cc5c03704a1ce47494961931f0263b0aac32505102595957531a2de69dd71aac51f8a49902f81f21283dbe8e21e01e5d82517868826f86acf338d935aa6b4d5a25c8d540389b277dd9d64569d68baf0f71bd03dba45b92a7fc052601d1bd011a2fc6790a23f97c6fa5caeea040ab86841f268d39ce4f7caf01069df78bba098e04366492f0c2ac24f1bf16828752765fa523c9a4d42b71109d123e6be8c7b1ab3ccf8ea03404075fe1a9596f1bba1d267f9a7879ceece514818316c9c0583469d2367831fc42b517ea028a28df7c18d783d16ea2436cee2b15d52db68b5dfdee6b4d26f0905f9b030c911a04d078923a4136afea96eed6874462a482917353264cc9bee298f167ac65a6db4e4eda88044b39cc0b33183843eaa946564a00c3a0ab661f2c915e70bf0bb65bfbb6fa2eea20aed16bf2c1a1d00ec55fb4ff2f76b8e462ea70c19efa579c9ee78194b86708fdae66a9ce6e2cf3d366037798cfb50277ba6d2fd4866361022fd788ab7735b40b8b61d55e32243e06719e53992e9ac16c9c4b6e6933635c3c47c8f7e73e17dd54d0dd8aeba5d76de46894e7b3f9d3ec25ad78ee82297ba69905ea0fa094b8667faa2b8885e2187b3da80268aa1164761d7b0d6de206b676777348152b8ae1d4afed753bc63c739a5ca8ce7afb2b241a226bd9e502baba391b5b13f5054f070b65a9cf3a67063bfaa803ba390732cd03888f664023f888741d04d564e0b5674b0a183ace81452001b3fbb4214c77d42ca75376742c471e58f67307726d56a1032bd236610cbcbcd03d0d7a452900136897dc55bb3ce959d10d4e6a10fb635006bd8c41cd9ded2d3dfdd8f2e229590324a7370cb2124210b2330f4c56155caa09a2564932ceded8d92c79664dcdeb87faad7d3da006cc2ea267ee3df41e9677789cc5a8cc3b83add6491561b3047919e0648b1b2e97d7ad6f6c2aa80cab8e9ae10e1f75b1fdd0246151af709d259a6a0ed0b26bd711024965ecad7c41387de45443defce53f66612948694a6032279131c257119ed876a8e805dfb49576ef5c563574115ee87050d92d191bc761ef51d966918e2ef925639400069e3959d8fe19f36136e947ff430bf74e71da0aa5923b00000000")
	v := natFromString("0x838332321d443a3d30373d47301d47073847473a383d3030f25b3d3d3e00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000002e00000000000000000041603038331c3d32f5303441e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e01c0a5459bfc7b9be9fcbb9d2383840464319434707303030f43a32f53034411c0a5459413820878787878787878787878787878787878787878787878787878787878787878787870630303a3a30334036605b923a6101f83638413943413960204337602043323801526040523241846038414143015238604060328452413841413638523c0240384141364036605b923a6101f83638413943413960204334602043323801526040523241846038414143015238604060328452413841413638523c02403841413638433030f25a8b83838383838383838383838383838383837d838383ffffffffffffffff838383838383838383000000000000000000030000007d26e27c7c8b83838383838383838383838383838383837d838383ffffffffffffffff83838383838383838383838383838383838383838383435960f535073030f3343200000000000000011881301938343030fa398383300000002300000000000000000000f11af4600c845252904141364138383c60406032414443095238010241414303364443434132305b595a15434160b042385341ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff47476043410536613603593a6005411c437405fcfcfcfcfcfcfc0000000000005a3b075815054359000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")
	q := nat(nil).make(16)
	q.div(stk, q, u, v)
}
