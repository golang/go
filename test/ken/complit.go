// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type	M	map[int]int
type	S	struct{ a,b,c int };
type	SS	struct{ aa,bb,cc S };
type	SA	struct{ a,b,c [3]int };
type	SC	struct{ a,b,c []int };
type	SM	struct{ a,b,c M };

func
main() {
	test("s.a", s.a);
	test("s.b", s.b);
	test("s.c", s.c);

	test("ss.aa.a", ss.aa.a);
	test("ss.aa.b", ss.aa.b);
	test("ss.aa.c", ss.aa.c);

	test("ss.bb.a", ss.bb.a);
	test("ss.bb.b", ss.bb.b);
	test("ss.bb.c", ss.bb.c);

	test("ss.cc.a", ss.cc.a);
	test("ss.cc.b", ss.cc.b);
	test("ss.cc.c", ss.cc.c);

	for i:=0; i<3; i++ {
		test("a[i]", a[i]);
		test("c[i]", c[i]);
		test("m[i]", m[i]);

		test("as[i].a", as[i].a);
		test("as[i].b", as[i].b);
		test("as[i].c", as[i].c);

		test("cs[i].a", cs[i].a);
		test("cs[i].b", cs[i].b);
		test("cs[i].c", cs[i].c);

		test("ms[i].a", ms[i].a);
		test("ms[i].b", ms[i].b);
		test("ms[i].c", ms[i].c);

		test("sa.a[i]", sa.a[i]);
		test("sa.b[i]", sa.b[i]);
		test("sa.c[i]", sa.c[i]);

		test("sc.a[i]", sc.a[i]);
		test("sc.b[i]", sc.b[i]);
		test("sc.c[i]", sc.c[i]);

		test("sm.a[i]", sm.a[i]);
		test("sm.b[i]", sm.b[i]);
		test("sm.c[i]", sm.c[i]);

		for j:=0; j<3; j++ {
			test("aa[i][j]", aa[i][j]);
			test("ac[i][j]", ac[i][j]);
			test("am[i][j]", am[i][j]);
			test("ca[i][j]", ca[i][j]);
			test("cc[i][j]", cc[i][j]);
			test("cm[i][j]", cm[i][j]);
			test("ma[i][j]", ma[i][j]);
			test("mc[i][j]", mc[i][j]);
			test("mm[i][j]", mm[i][j]);
		}
	}

}

var	ref	= 0;

func
test(xs string, x int) {

	if ref >= len(answers) {
		println(xs, x);
		return;
	}

	if x != answers[ref] {
		println(xs, "is", x, "should be", answers[ref])
	}
	ref++;
}


var	a	= [3]int{1001, 1002, 1003}
var	s	= S{1101, 1102, 1103}
var	c	= []int{1201, 1202, 1203}
var	m	= M{0:1301, 1:1302, 2:1303}

var	aa	= [3][3]int{[3]int{2001,2002,2003}, [3]int{2004,2005,2006}, [3]int{2007,2008,2009}}
var	as	= [3]S{S{2101,2102,2103},S{2104,2105,2106},S{2107,2108,2109}}
var	ac	= [3][]int{[]int{2201,2202,2203}, []int{2204,2205,2206}, []int{2207,2208,2209}}
var	am	= [3]M{M{0:2301,1:2302,2:2303}, M{0:2304,1:2305,2:2306}, M{0:2307,1:2308,2:2309}}

var	sa	= SA{[3]int{3001,3002,3003},[3]int{3004,3005,3006},[3]int{3007,3008,3009}}
var	ss	= SS{S{3101,3102,3103},S{3104,3105,3106},S{3107,3108,3109}}
var	sc	= SC{[]int{3201,3202,3203},[]int{3204,3205,3206},[]int{3207,3208,3209}}
var	sm	= SM{M{0:3301,1:3302,2:3303}, M{0:3304,1:3305,2:3306}, M{0:3307,1:3308,2:3309}}

var	ca	= [][3]int{[3]int{4001,4002,4003}, [3]int{4004,4005,4006}, [3]int{4007,4008,4009}}
var	cs	= []S{S{4101,4102,4103},S{4104,4105,4106},S{4107,4108,4109}}
var	cc	= [][]int{[]int{4201,4202,4203}, []int{4204,4205,4206}, []int{4207,4208,4209}}
var	cm	= []M{M{0:4301,1:4302,2:4303}, M{0:4304,1:4305,2:4306}, M{0:4307,1:4308,2:4309}}

var	ma	= map[int][3]int{0:[3]int{5001,5002,5003}, 1:[3]int{5004,5005,5006}, 2:[3]int{5007,5008,5009}}
var	ms	= map[int]S{0:S{5101,5102,5103},1:S{5104,5105,5106},2:S{5107,5108,5109}}
var	mc	= map[int][]int{0:[]int{5201,5202,5203}, 1:[]int{5204,5205,5206}, 2:[]int{5207,5208,5209}}
var	mm	= map[int]M{0:M{0:5301,1:5302,2:5303}, 1:M{0:5304,1:5305,2:5306}, 2:M{0:5307,1:5308,2:5309}}

var	answers	= [...]int {
	// s
	1101, 1102, 1103,

	// ss
	3101, 3102, 3103,
	3104, 3105, 3106,
	3107, 3108, 3109,

	// [0]
	1001, 1201, 1301,
	2101, 2102, 2103,
	4101, 4102, 4103,
	5101, 5102, 5103,
	3001, 3004, 3007,
	3201, 3204, 3207,
	3301, 3304, 3307,

	// [0][j]
	2001, 2201, 2301, 4001, 4201, 4301, 5001, 5201, 5301,
	2002, 2202, 2302, 4002, 4202, 4302, 5002, 5202, 5302,
	2003, 2203, 2303, 4003, 4203, 4303, 5003, 5203, 5303,

	// [1]
	1002, 1202, 1302,
	2104, 2105, 2106,
	4104, 4105, 4106,
	5104, 5105, 5106,
	3002, 3005, 3008,
	3202, 3205, 3208,
	3302, 3305, 3308,

	// [1][j]
	2004, 2204, 2304, 4004, 4204, 4304, 5004, 5204, 5304,
	2005, 2205, 2305, 4005, 4205, 4305, 5005, 5205, 5305,
	2006, 2206, 2306, 4006, 4206, 4306, 5006, 5206, 5306,

	// [2]
	1003, 1203, 1303,
	2107, 2108, 2109,
	4107, 4108, 4109,
	5107, 5108, 5109,
	3003, 3006, 3009,
	3203, 3206, 3209,
	3303, 3306, 3309,

	// [2][j]
	2007, 2207, 2307, 4007, 4207, 4307, 5007, 5207, 5307,
	2008, 2208, 2308, 4008, 4208, 4308, 5008, 5208, 5308,
	2009, 2209, 2309, 4009, 4209, 4309, 5009, 5209, 5309,
}
