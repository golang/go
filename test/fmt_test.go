// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// $G $F.go && $L $F.$A && ./$A.out

package main

import fmt "fmt"  // BUG: shouldn't need the first 'fmt'.

func E(f *fmt.Fmt, e string) {
	g := f.str();
	if sys.argc() > 1 {
		print(g, "\n");
		if g != e {
			print("expected <", e, ">\n");
		}
		return;
	}
	if g != e {
		print("expected <", e, ">\n");
		print("got      <", g, ">\n");
	}
}

const B32 = 1<<32 - 1
const B64 = 1<<64 - 1

func main() {
	f := fmt.New();
	E(f.s("d   ").d(1234), "d   1234");
	E(f.s("Simple output\n"), "Simple output\n");
	E(f.s("\td   ").d(-1234), "\td   -1234");
	E(f.s("\tud  ").ud(B32), "\tud  4294967295");
	E(f.s("\tud64  ").ud64(B64), "\tud64  18446744073709551615");
	E(f.s("\to   ").o(01234), "\to   1234");
	E(f.s("\tuo  ").uo(B32), "\tuo  37777777777");
	E(f.s("\tuo64  ").uo64(B64), "\tuo64  1777777777777777777777");
	E(f.s("\tx   ").x(0x1234abcd), "\tx   1234abcd");
	E(f.s("\tux  ").ux(B32 - 0x01234567), "\tux  fedcba98");
	E(f.s("\tX  ").X(0x1234abcd), "\tX  1234ABCD");
	E(f.s("\tuX ").uX(B32 - 0x01234567), "\tuX FEDCBA98");
	E(f.s("\tux64  ").ux64(B64), "\tux64  ffffffffffffffff");
	E(f.s("\tb   ").b(7), "\tb   111");
	E(f.s("\tb64   ").b64(B64), "\tb64   1111111111111111111111111111111111111111111111111111111111111111");
	E(f.s("\te   ").e64(1.), "\te   1.000000e+00");
	E(f.s("\te   ").e64(1234.5678e3), "\te   1.234567e+06");
	E(f.s("\te   ").e64(1234.5678e-8), "\te   1.234567e-05");
	E(f.s("\te   ").e64(-7.0), "\te   -7.000000e+00");
	E(f.s("\te   ").e64(-1e-9), "\te   -1.000000e-09");
	E(f.s("\tf   ").f64(1234.5678e3), "\tf   1234567.800000");
	E(f.s("\tf   ").f64(1234.5678e-8), "\tf   0.000012");
	E(f.s("\tf   ").f64(-7.0), "\tf   -7.000000");
	E(f.s("\tf   ").f64(-1e-9), "\tf   -0.000000");
	E(f.s("\tg   ").g64(1234.5678e3), "\tg   1234567.8");
	E(f.s("\tg   ").g64(1234.5678e-8), "\tg   0.000012");
	E(f.s("\tg   ").g64(-7.0), "\tg   -7.");
	E(f.s("\tg   ").g64(-1e-9), "\tg   -0.");
	E(f.s("\tc   ").c('x'), "\tc   x");
	E(f.s("\tc   ").c(0xe4), "\tc   ä");
	E(f.s("\tc   ").c(0x672c), "\tc   本");
	E(f.s("\tc   ").c('日'), "\tc   日");

	E(f.s("Flags, width, and precision"), "Flags, width, and precision");
	E(f.s("\t\t|123456789_123456789_"), "\t\t|123456789_123456789_");
	E(f.s("\t20.8d\t|").wp(20,8).d(1234).s("|"), "\t20.8d\t|            00001234|");
	E(f.s("\t20.8d\t|").wp(20,8).d(-1234).s("|"), "\t20.8d\t|           -00001234|");
	E(f.s("\t20d\t|").w(20).d(1234).s("|"), "\t20d\t|                1234|");
	E(f.s("\t-20.8d\t|").wp(-20,8).d(1234).s("|"), "\t-20.8d\t|00001234            |");
	E(f.s("\t-20.8d\t|").wp(-20,8).d(-1234).s("|"), "\t-20.8d\t|-00001234           |");
	E(f.s("\t.20b\t|").p(20).b(7).s("|"), "\t.20b\t|00000000000000000111|");
	E(f.s("\t20.5s\t|").wp(20,5).s("qwertyuiop").s("|"), "\t20.5s\t|               qwert|");
	E(f.s("\t.5s\t|").p(5).s("qwertyuiop").s("|"), "\t.5s\t|qwert|");
	E(f.s("\t-20.5s\t|").wp(-20,5).s("qwertyuiop").s("|"), "\t-20.5s\t|qwert               |");
	E(f.s("\t20c\t|").w(20).c('x').s("|"), "\t20c\t|                   x|");
	E(f.s("\t-20c\t|").w(-20).c('x').s("|"), "\t-20c\t|x                   |");
	E(f.s("\t20e\t|").w(20).e(1.2345e3).s("|"), "\t20e\t|        1.234500e+03|");
	E(f.s("\t20e\t|").w(20).e(1.2345e-3).s("|"), "\t20e\t|        1.234500e-03|");
	E(f.s("\t-20e\t|").w(-20).e(1.2345e3).s("|"), "\t-20e\t|1.234500e+03        |");
	E(f.s("\t20.8e\t|").wp(20,8).e(1.2345e3).s("|"), "\t20.8e\t|      1.23450000e+03|");
	E(f.s("\t20f\t|").w(20).f64(1.23456789e3).s("|"), "\t20f\t|         1234.567890|");
	E(f.s("\t20f\t|").w(20).f64(1.23456789e-3).s("|"), "\t20f\t|            0.001235|");
	E(f.s("\t20f\t|").w(20).f64(12345678901.23456789).s("|"), "\t20f\t|  12345678901.234570|");
	E(f.s("\t-20f\t|").w(-20).f64(1.23456789e3).s("|"), "\t-20f\t|1234.567890         |");
	E(f.s("\t20.8f\t|").wp(20,8).f64(1.23456789e3).s("|"), "\t20.8f\t|       1234.56789000|");
	E(f.s("\t20.8f\t|").wp(20,8).f64(1.23456789e-3).s("|"), "\t20.8f\t|          0.00123457|");
	E(f.s("\tg\t|").g64(1.23456789e3).s("|"), "\tg\t|1234.56789|");
	E(f.s("\tg\t|").g64(1.23456789e-3).s("|"), "\tg\t|0.001235|");
	E(f.s("\tg\t|").g64(1.23456789e20).s("|"), "\tg\t|1.234567e+20|");

	E(f.s("\tE\t|").w(20).g64(sys.Inf(1)).s("|"), "\tE\t|                 Inf|");
	E(f.s("\tF\t|").w(-20).g64(sys.Inf(-1)).s("|"), "\tF\t|-Inf                |");
	E(f.s("\tG\t|").w(20).g64(sys.NaN()).s("|"), "\tG\t|                 NaN|");
}
