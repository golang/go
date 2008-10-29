// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import Bignum "bignum"

const (
	sa = "991";
	sb = "2432902008176640000";  // 20!
	sc = "93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000";  // 100!
)


var (
	a = Bignum.NatFromString(sa, 10);
	b = Bignum.NatFromString(sb, 10);
	c = Bignum.NatFromString(sc, 10);
)


func TEST(msg string, b bool) {
	if !b {
		panic("TEST failed: ", msg, "\n");
	}
}


func TestConv() {
	TEST("TC1", a.Cmp(Bignum.NewNat(991)) == 0);
	TEST("TC2", b.Cmp(Bignum.Fact(20)) == 0);
	TEST("TC3", c.Cmp(Bignum.Fact(100)) == 0);
	TEST("TC4", a.String(10) == sa);
	TEST("TC5", b.String(10) == sb);
	TEST("TC6", c.String(10) == sc);
}


func main() {
	TestConv();
	print("PASSED\n");
}
