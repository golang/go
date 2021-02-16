package test

import "testing"

// Tests to make sure logic simplification rules are correct.

func TestLogic64(t *testing.T) {
	// test values to determine function equality
	values := [...]int64{-1 << 63, 1<<63 - 1, -4, -3, -2, -1, 0, 1, 2, 3, 4}

	// golden functions we use repeatedly
	zero := func(x int64) int64 { return 0 }
	id := func(x int64) int64 { return x }
	or := func(x, y int64) int64 { return x | y }
	and := func(x, y int64) int64 { return x & y }
	y := func(x, y int64) int64 { return y }

	for _, test := range [...]struct {
		name   string
		f      func(int64) int64
		golden func(int64) int64
	}{
		{"x|x", func(x int64) int64 { return x | x }, id},
		{"x|0", func(x int64) int64 { return x | 0 }, id},
		{"x|-1", func(x int64) int64 { return x | -1 }, func(x int64) int64 { return -1 }},
		{"x&x", func(x int64) int64 { return x & x }, id},
		{"x&0", func(x int64) int64 { return x & 0 }, zero},
		{"x&-1", func(x int64) int64 { return x & -1 }, id},
		{"x^x", func(x int64) int64 { return x ^ x }, zero},
		{"x^0", func(x int64) int64 { return x ^ 0 }, id},
		{"x^-1", func(x int64) int64 { return x ^ -1 }, func(x int64) int64 { return ^x }},
		{"x+0", func(x int64) int64 { return x + 0 }, id},
		{"x-x", func(x int64) int64 { return x - x }, zero},
		{"x*0", func(x int64) int64 { return x * 0 }, zero},
		{"^^x", func(x int64) int64 { return ^^x }, id},
	} {
		for _, v := range values {
			got := test.f(v)
			want := test.golden(v)
			if want != got {
				t.Errorf("[%s](%d)=%d, want %d", test.name, v, got, want)
			}
		}
	}
	for _, test := range [...]struct {
		name   string
		f      func(int64, int64) int64
		golden func(int64, int64) int64
	}{
		{"x|(x|y)", func(x, y int64) int64 { return x | (x | y) }, or},
		{"x|(y|x)", func(x, y int64) int64 { return x | (y | x) }, or},
		{"(x|y)|x", func(x, y int64) int64 { return (x | y) | x }, or},
		{"(y|x)|x", func(x, y int64) int64 { return (y | x) | x }, or},
		{"x&(x&y)", func(x, y int64) int64 { return x & (x & y) }, and},
		{"x&(y&x)", func(x, y int64) int64 { return x & (y & x) }, and},
		{"(x&y)&x", func(x, y int64) int64 { return (x & y) & x }, and},
		{"(y&x)&x", func(x, y int64) int64 { return (y & x) & x }, and},
		{"x^(x^y)", func(x, y int64) int64 { return x ^ (x ^ y) }, y},
		{"x^(y^x)", func(x, y int64) int64 { return x ^ (y ^ x) }, y},
		{"(x^y)^x", func(x, y int64) int64 { return (x ^ y) ^ x }, y},
		{"(y^x)^x", func(x, y int64) int64 { return (y ^ x) ^ x }, y},
		{"-(y-x)", func(x, y int64) int64 { return -(y - x) }, func(x, y int64) int64 { return x - y }},
		{"(x+y)-x", func(x, y int64) int64 { return (x + y) - x }, y},
		{"(y+x)-x", func(x, y int64) int64 { return (y + x) - x }, y},
	} {
		for _, v := range values {
			for _, w := range values {
				got := test.f(v, w)
				want := test.golden(v, w)
				if want != got {
					t.Errorf("[%s](%d,%d)=%d, want %d", test.name, v, w, got, want)
				}
			}
		}
	}
}

func TestLogic32(t *testing.T) {
	// test values to determine function equality
	values := [...]int32{-1 << 31, 1<<31 - 1, -4, -3, -2, -1, 0, 1, 2, 3, 4}

	// golden functions we use repeatedly
	zero := func(x int32) int32 { return 0 }
	id := func(x int32) int32 { return x }
	or := func(x, y int32) int32 { return x | y }
	and := func(x, y int32) int32 { return x & y }
	y := func(x, y int32) int32 { return y }

	for _, test := range [...]struct {
		name   string
		f      func(int32) int32
		golden func(int32) int32
	}{
		{"x|x", func(x int32) int32 { return x | x }, id},
		{"x|0", func(x int32) int32 { return x | 0 }, id},
		{"x|-1", func(x int32) int32 { return x | -1 }, func(x int32) int32 { return -1 }},
		{"x&x", func(x int32) int32 { return x & x }, id},
		{"x&0", func(x int32) int32 { return x & 0 }, zero},
		{"x&-1", func(x int32) int32 { return x & -1 }, id},
		{"x^x", func(x int32) int32 { return x ^ x }, zero},
		{"x^0", func(x int32) int32 { return x ^ 0 }, id},
		{"x^-1", func(x int32) int32 { return x ^ -1 }, func(x int32) int32 { return ^x }},
		{"x+0", func(x int32) int32 { return x + 0 }, id},
		{"x-x", func(x int32) int32 { return x - x }, zero},
		{"x*0", func(x int32) int32 { return x * 0 }, zero},
		{"^^x", func(x int32) int32 { return ^^x }, id},
	} {
		for _, v := range values {
			got := test.f(v)
			want := test.golden(v)
			if want != got {
				t.Errorf("[%s](%d)=%d, want %d", test.name, v, got, want)
			}
		}
	}
	for _, test := range [...]struct {
		name   string
		f      func(int32, int32) int32
		golden func(int32, int32) int32
	}{
		{"x|(x|y)", func(x, y int32) int32 { return x | (x | y) }, or},
		{"x|(y|x)", func(x, y int32) int32 { return x | (y | x) }, or},
		{"(x|y)|x", func(x, y int32) int32 { return (x | y) | x }, or},
		{"(y|x)|x", func(x, y int32) int32 { return (y | x) | x }, or},
		{"x&(x&y)", func(x, y int32) int32 { return x & (x & y) }, and},
		{"x&(y&x)", func(x, y int32) int32 { return x & (y & x) }, and},
		{"(x&y)&x", func(x, y int32) int32 { return (x & y) & x }, and},
		{"(y&x)&x", func(x, y int32) int32 { return (y & x) & x }, and},
		{"x^(x^y)", func(x, y int32) int32 { return x ^ (x ^ y) }, y},
		{"x^(y^x)", func(x, y int32) int32 { return x ^ (y ^ x) }, y},
		{"(x^y)^x", func(x, y int32) int32 { return (x ^ y) ^ x }, y},
		{"(y^x)^x", func(x, y int32) int32 { return (y ^ x) ^ x }, y},
		{"-(y-x)", func(x, y int32) int32 { return -(y - x) }, func(x, y int32) int32 { return x - y }},
		{"(x+y)-x", func(x, y int32) int32 { return (x + y) - x }, y},
		{"(y+x)-x", func(x, y int32) int32 { return (y + x) - x }, y},
	} {
		for _, v := range values {
			for _, w := range values {
				got := test.f(v, w)
				want := test.golden(v, w)
				if want != got {
					t.Errorf("[%s](%d,%d)=%d, want %d", test.name, v, w, got, want)
				}
			}
		}
	}
}

func TestLogic16(t *testing.T) {
	// test values to determine function equality
	values := [...]int16{-1 << 15, 1<<15 - 1, -4, -3, -2, -1, 0, 1, 2, 3, 4}

	// golden functions we use repeatedly
	zero := func(x int16) int16 { return 0 }
	id := func(x int16) int16 { return x }
	or := func(x, y int16) int16 { return x | y }
	and := func(x, y int16) int16 { return x & y }
	y := func(x, y int16) int16 { return y }

	for _, test := range [...]struct {
		name   string
		f      func(int16) int16
		golden func(int16) int16
	}{
		{"x|x", func(x int16) int16 { return x | x }, id},
		{"x|0", func(x int16) int16 { return x | 0 }, id},
		{"x|-1", func(x int16) int16 { return x | -1 }, func(x int16) int16 { return -1 }},
		{"x&x", func(x int16) int16 { return x & x }, id},
		{"x&0", func(x int16) int16 { return x & 0 }, zero},
		{"x&-1", func(x int16) int16 { return x & -1 }, id},
		{"x^x", func(x int16) int16 { return x ^ x }, zero},
		{"x^0", func(x int16) int16 { return x ^ 0 }, id},
		{"x^-1", func(x int16) int16 { return x ^ -1 }, func(x int16) int16 { return ^x }},
		{"x+0", func(x int16) int16 { return x + 0 }, id},
		{"x-x", func(x int16) int16 { return x - x }, zero},
		{"x*0", func(x int16) int16 { return x * 0 }, zero},
		{"^^x", func(x int16) int16 { return ^^x }, id},
	} {
		for _, v := range values {
			got := test.f(v)
			want := test.golden(v)
			if want != got {
				t.Errorf("[%s](%d)=%d, want %d", test.name, v, got, want)
			}
		}
	}
	for _, test := range [...]struct {
		name   string
		f      func(int16, int16) int16
		golden func(int16, int16) int16
	}{
		{"x|(x|y)", func(x, y int16) int16 { return x | (x | y) }, or},
		{"x|(y|x)", func(x, y int16) int16 { return x | (y | x) }, or},
		{"(x|y)|x", func(x, y int16) int16 { return (x | y) | x }, or},
		{"(y|x)|x", func(x, y int16) int16 { return (y | x) | x }, or},
		{"x&(x&y)", func(x, y int16) int16 { return x & (x & y) }, and},
		{"x&(y&x)", func(x, y int16) int16 { return x & (y & x) }, and},
		{"(x&y)&x", func(x, y int16) int16 { return (x & y) & x }, and},
		{"(y&x)&x", func(x, y int16) int16 { return (y & x) & x }, and},
		{"x^(x^y)", func(x, y int16) int16 { return x ^ (x ^ y) }, y},
		{"x^(y^x)", func(x, y int16) int16 { return x ^ (y ^ x) }, y},
		{"(x^y)^x", func(x, y int16) int16 { return (x ^ y) ^ x }, y},
		{"(y^x)^x", func(x, y int16) int16 { return (y ^ x) ^ x }, y},
		{"-(y-x)", func(x, y int16) int16 { return -(y - x) }, func(x, y int16) int16 { return x - y }},
		{"(x+y)-x", func(x, y int16) int16 { return (x + y) - x }, y},
		{"(y+x)-x", func(x, y int16) int16 { return (y + x) - x }, y},
	} {
		for _, v := range values {
			for _, w := range values {
				got := test.f(v, w)
				want := test.golden(v, w)
				if want != got {
					t.Errorf("[%s](%d,%d)=%d, want %d", test.name, v, w, got, want)
				}
			}
		}
	}
}

func TestLogic8(t *testing.T) {
	// test values to determine function equality
	values := [...]int8{-1 << 7, 1<<7 - 1, -4, -3, -2, -1, 0, 1, 2, 3, 4}

	// golden functions we use repeatedly
	zero := func(x int8) int8 { return 0 }
	id := func(x int8) int8 { return x }
	or := func(x, y int8) int8 { return x | y }
	and := func(x, y int8) int8 { return x & y }
	y := func(x, y int8) int8 { return y }

	for _, test := range [...]struct {
		name   string
		f      func(int8) int8
		golden func(int8) int8
	}{
		{"x|x", func(x int8) int8 { return x | x }, id},
		{"x|0", func(x int8) int8 { return x | 0 }, id},
		{"x|-1", func(x int8) int8 { return x | -1 }, func(x int8) int8 { return -1 }},
		{"x&x", func(x int8) int8 { return x & x }, id},
		{"x&0", func(x int8) int8 { return x & 0 }, zero},
		{"x&-1", func(x int8) int8 { return x & -1 }, id},
		{"x^x", func(x int8) int8 { return x ^ x }, zero},
		{"x^0", func(x int8) int8 { return x ^ 0 }, id},
		{"x^-1", func(x int8) int8 { return x ^ -1 }, func(x int8) int8 { return ^x }},
		{"x+0", func(x int8) int8 { return x + 0 }, id},
		{"x-x", func(x int8) int8 { return x - x }, zero},
		{"x*0", func(x int8) int8 { return x * 0 }, zero},
		{"^^x", func(x int8) int8 { return ^^x }, id},
	} {
		for _, v := range values {
			got := test.f(v)
			want := test.golden(v)
			if want != got {
				t.Errorf("[%s](%d)=%d, want %d", test.name, v, got, want)
			}
		}
	}
	for _, test := range [...]struct {
		name   string
		f      func(int8, int8) int8
		golden func(int8, int8) int8
	}{
		{"x|(x|y)", func(x, y int8) int8 { return x | (x | y) }, or},
		{"x|(y|x)", func(x, y int8) int8 { return x | (y | x) }, or},
		{"(x|y)|x", func(x, y int8) int8 { return (x | y) | x }, or},
		{"(y|x)|x", func(x, y int8) int8 { return (y | x) | x }, or},
		{"x&(x&y)", func(x, y int8) int8 { return x & (x & y) }, and},
		{"x&(y&x)", func(x, y int8) int8 { return x & (y & x) }, and},
		{"(x&y)&x", func(x, y int8) int8 { return (x & y) & x }, and},
		{"(y&x)&x", func(x, y int8) int8 { return (y & x) & x }, and},
		{"x^(x^y)", func(x, y int8) int8 { return x ^ (x ^ y) }, y},
		{"x^(y^x)", func(x, y int8) int8 { return x ^ (y ^ x) }, y},
		{"(x^y)^x", func(x, y int8) int8 { return (x ^ y) ^ x }, y},
		{"(y^x)^x", func(x, y int8) int8 { return (y ^ x) ^ x }, y},
		{"-(y-x)", func(x, y int8) int8 { return -(y - x) }, func(x, y int8) int8 { return x - y }},
		{"(x+y)-x", func(x, y int8) int8 { return (x + y) - x }, y},
		{"(y+x)-x", func(x, y int8) int8 { return (y + x) - x }, y},
	} {
		for _, v := range values {
			for _, w := range values {
				got := test.f(v, w)
				want := test.golden(v, w)
				if want != got {
					t.Errorf("[%s](%d,%d)=%d, want %d", test.name, v, w, got, want)
				}
			}
		}
	}
}
