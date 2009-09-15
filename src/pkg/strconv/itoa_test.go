// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv_test

import (
	"fmt";
	"os";
	. "strconv";
	"testing";
)

type itob64Test struct {
	in int64;
	base uint;
	out string;
}

var itob64tests = []itob64Test {
	itob64Test{ 0, 10, "0" },
	itob64Test{ 1, 10, "1" },
	itob64Test{ -1, 10, "-1" },
	itob64Test{ 12345678, 10, "12345678" },
	itob64Test{ -987654321, 10, "-987654321" },
	itob64Test{ 1<<31-1, 10, "2147483647" },
	itob64Test{ -1<<31+1, 10, "-2147483647" },
	itob64Test{ 1<<31, 10, "2147483648" },
	itob64Test{ -1<<31, 10, "-2147483648" },
	itob64Test{ 1<<31+1, 10, "2147483649" },
	itob64Test{ -1<<31-1, 10, "-2147483649" },
	itob64Test{ 1<<32-1, 10, "4294967295" },
	itob64Test{ -1<<32+1, 10, "-4294967295" },
	itob64Test{ 1<<32, 10, "4294967296" },
	itob64Test{ -1<<32, 10, "-4294967296" },
	itob64Test{ 1<<32+1, 10, "4294967297" },
	itob64Test{ -1<<32-1, 10, "-4294967297" },
	itob64Test{ 1<<50, 10, "1125899906842624" },
	itob64Test{ 1<<63-1, 10, "9223372036854775807" },
	itob64Test{ -1<<63+1, 10, "-9223372036854775807" },
	itob64Test{ -1<<63, 10, "-9223372036854775808" },

	itob64Test{ 0, 2, "0" },
	itob64Test{ 10, 2, "1010" },
	itob64Test{ -1, 2, "-1" },
	itob64Test{ 1<<15, 2, "1000000000000000" },

	itob64Test{ -8, 8, "-10" },
	itob64Test{ 057635436545, 8, "57635436545" },
	itob64Test{ 1<<24, 8, "100000000" },

	itob64Test{ 16, 16, "10" },
	itob64Test{ -0x123456789abcdef, 16, "-123456789abcdef" },
	itob64Test{ 1<<63-1, 16, "7fffffffffffffff" },

	itob64Test{ 16, 17, "g" },
	itob64Test{ 25, 25, "10" },
	itob64Test{ (((((17*35+24)*35+21)*35+34)*35+12)*35+24)*35+32, 35, "holycow" },
	itob64Test{ (((((17*36+24)*36+21)*36+34)*36+12)*36+24)*36+32, 36, "holycow" },
}

func TestItoa(t *testing.T) {
	for _, test := range itob64tests {
		s := Itob64(test.in, test.base);
		if s != test.out {
			t.Errorf("Itob64(%v, %v) = %v want %v\n",
				test.in, test.base, s, test.out);
		}

		if test.in >= 0 {
			s := Uitob64(uint64(test.in), test.base);
			if s != test.out {
				t.Errorf("Uitob64(%v, %v) = %v want %v\n",
					test.in, test.base, s, test.out);
			}
		}

		if int64(int(test.in)) == test.in {
			s := Itob(int(test.in), test.base);
			if s != test.out {
				t.Errorf("Itob(%v, %v) = %v want %v\n",
					test.in, test.base, s, test.out);
			}

			if test.in >= 0  {
				s := Uitob(uint(test.in), test.base);
				if s != test.out {
					t.Errorf("Uitob(%v, %v) = %v want %v\n",
						test.in, test.base, s, test.out);
				}
			}
		}

		if test.base == 10 {
			s := Itoa64(test.in);
			if s != test.out {
				t.Errorf("Itoa64(%v) = %v want %v\n",
					test.in, s, test.out);
			}

			if test.in >= 0 {
				s := Uitob64(uint64(test.in), test.base);
				if s != test.out {
					t.Errorf("Uitob64(%v, %v) = %v want %v\n",
						test.in, test.base, s, test.out);
				}
			}

			if int64(int(test.in)) == test.in {
				s := Itoa(int(test.in));
				if s != test.out {
					t.Errorf("Itoa(%v) = %v want %v\n",
						test.in, s, test.out);
				}

				if test.in >= 0 {
					s := Uitoa(uint(test.in));
					if s != test.out {
						t.Errorf("Uitoa(%v) = %v want %v\n",
							test.in, s, test.out);
					}
				}
			}
		}
	}
}

type uitob64Test struct {
	in uint64;
	base uint;
	out string;
}

var uitob64tests = []uitob64Test {
	uitob64Test{ 1<<63-1, 10, "9223372036854775807" },
	uitob64Test{ 1<<63, 10, "9223372036854775808" },
	uitob64Test{ 1<<63+1, 10, "9223372036854775809" },
	uitob64Test{ 1<<64-2, 10, "18446744073709551614" },
	uitob64Test{ 1<<64-1, 10, "18446744073709551615" },
}

func TestUitoa(t *testing.T) {
	for _, test := range uitob64tests {
		s := Uitob64(test.in, test.base);
		if s != test.out {
			t.Errorf("Uitob64(%v, %v) = %v want %v\n",
				test.in, test.base, s, test.out);
		}

		if uint64(uint(test.in)) == test.in {
			s := Uitob(uint(test.in), test.base);
			if s != test.out {
				t.Errorf("Uitob(%v, %v) = %v want %v\n",
					test.in, test.base, s, test.out);
			}
		}

		if test.base == 10 {
			s := Uitoa64(test.in);
			if s != test.out {
				t.Errorf("Uitoa64(%v) = %v want %v\n",
					test.in, s, test.out);
			}

			if uint64(uint(test.in)) == test.in {
				s := Uitoa(uint(test.in));
				if s != test.out {
					t.Errorf("Uitoa(%v) = %v want %v\n",
						test.in, s, test.out);
				}
			}
		}
	}
}
