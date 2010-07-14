// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eval

import (
	"big"
	"testing"
)

var undefined = "undefined"
var typeAsExpr = "type .* used as expression"
var badCharLit = "character literal"
var unknownEscape = "unknown escape sequence"
var opTypes = "illegal (operand|argument) type|cannot index into"
var badAddrOf = "cannot take the address"
var constantTruncated = "constant [^ ]* truncated"
var constantUnderflows = "constant [^ ]* underflows"
var constantOverflows = "constant [^ ]* overflows"
var implLimit = "implementation limit"
var mustBeUnsigned = "must be unsigned"
var divByZero = "divide by zero"

var hugeInteger = new(big.Int).Lsh(idealOne, 64)

var exprTests = []test{
	Val("i", 1),
	CErr("zzz", undefined),
	// TODO(austin) Test variable in constant context
	//CErr("t", typeAsExpr),

	Val("'a'", big.NewInt('a')),
	Val("'\\uffff'", big.NewInt('\uffff')),
	Val("'\\n'", big.NewInt('\n')),
	CErr("''+x", badCharLit),
	// Produces two parse errors
	//CErr("'''", ""),
	CErr("'\n'", badCharLit),
	CErr("'\\z'", unknownEscape),
	CErr("'ab'", badCharLit),

	Val("1.0", big.NewRat(1, 1)),
	Val("1.", big.NewRat(1, 1)),
	Val(".1", big.NewRat(1, 10)),
	Val("1e2", big.NewRat(100, 1)),

	Val("\"abc\"", "abc"),
	Val("\"\"", ""),
	Val("\"\\n\\\"\"", "\n\""),
	CErr("\"\\z\"", unknownEscape),
	CErr("\"abc", "string not terminated"),

	Val("(i)", 1),

	Val("ai[0]", 1),
	Val("(&ai)[0]", 1),
	Val("ai[1]", 2),
	Val("ai[i]", 2),
	Val("ai[u]", 2),
	CErr("ai[f]", opTypes),
	CErr("ai[0][0]", opTypes),
	CErr("ai[2]", "index 2 exceeds"),
	CErr("ai[1+1]", "index 2 exceeds"),
	CErr("ai[-1]", "negative index"),
	RErr("ai[i+i]", "index 2 exceeds"),
	RErr("ai[-i]", "negative index"),
	CErr("i[0]", opTypes),
	CErr("f[0]", opTypes),

	Val("aai[0][0]", 1),
	Val("aai[1][1]", 4),
	CErr("aai[2][0]", "index 2 exceeds"),
	CErr("aai[0][2]", "index 2 exceeds"),

	Val("sli[0]", 1),
	Val("sli[1]", 2),
	CErr("sli[-1]", "negative index"),
	RErr("sli[-i]", "negative index"),
	RErr("sli[2]", "index 2 exceeds"),

	Val("s[0]", uint8('a')),
	Val("s[1]", uint8('b')),
	CErr("s[-1]", "negative index"),
	RErr("s[-i]", "negative index"),
	RErr("s[3]", "index 3 exceeds"),

	Val("ai[0:2]", vslice{varray{1, 2}, 2, 2}),
	Val("ai[0:1]", vslice{varray{1, 2}, 1, 2}),
	Val("ai[0:]", vslice{varray{1, 2}, 2, 2}),
	Val("ai[i:]", vslice{varray{2}, 1, 1}),

	Val("sli[0:2]", vslice{varray{1, 2, 3}, 2, 3}),
	Val("sli[0:i]", vslice{varray{1, 2, 3}, 1, 3}),
	Val("sli[1:]", vslice{varray{2, 3}, 1, 2}),

	CErr("1(2)", "cannot call"),
	CErr("fn(1,2)", "too many"),
	CErr("fn()", "not enough"),
	CErr("fn(true)", opTypes),
	CErr("fn(true)", "function call"),
	// Single argument functions don't say which argument.
	//CErr("fn(true)", "argument 1"),
	Val("fn(1)", 2),
	Val("fn(1.0)", 2),
	CErr("fn(1.5)", constantTruncated),
	Val("fn(i)", 2),
	CErr("fn(u)", opTypes),

	CErr("void()+2", opTypes),
	CErr("oneTwo()+2", opTypes),

	Val("cap(ai)", 2),
	Val("cap(&ai)", 2),
	Val("cap(aai)", 2),
	Val("cap(sli)", 3),
	CErr("cap(0)", opTypes),
	CErr("cap(i)", opTypes),
	CErr("cap(s)", opTypes),

	Val("len(s)", 3),
	Val("len(ai)", 2),
	Val("len(&ai)", 2),
	Val("len(ai[0:])", 2),
	Val("len(ai[1:])", 1),
	Val("len(ai[2:])", 0),
	Val("len(aai)", 2),
	Val("len(sli)", 2),
	Val("len(sli[0:])", 2),
	Val("len(sli[1:])", 1),
	Val("len(sli[2:])", 0),
	// TODO(austin) Test len of map
	CErr("len(0)", opTypes),
	CErr("len(i)", opTypes),

	CErr("*i", opTypes),
	Val("*&i", 1),
	Val("*&(i)", 1),
	CErr("&1", badAddrOf),
	CErr("&c", badAddrOf),
	Val("*(&ai[0])", 1),

	Val("+1", big.NewInt(+1)),
	Val("+1.0", big.NewRat(1, 1)),
	Val("01.5", big.NewRat(15, 10)),
	CErr("+\"x\"", opTypes),

	Val("-42", big.NewInt(-42)),
	Val("-i", -1),
	Val("-f", -1.0),
	// 6g bug?
	//Val("-(f-1)", -0.0),
	CErr("-\"x\"", opTypes),

	// TODO(austin) Test unary !

	Val("^2", big.NewInt(^2)),
	Val("^(-2)", big.NewInt(^(-2))),
	CErr("^2.0", opTypes),
	CErr("^2.5", opTypes),
	Val("^i", ^1),
	Val("^u", ^uint(1)),
	CErr("^f", opTypes),

	Val("1+i", 2),
	Val("1+u", uint(2)),
	Val("3.0+i", 4),
	Val("1+1", big.NewInt(2)),
	Val("f+f", 2.0),
	Val("1+f", 2.0),
	Val("1.0+1", big.NewRat(2, 1)),
	Val("\"abc\" + \"def\"", "abcdef"),
	CErr("i+u", opTypes),
	CErr("-1+u", constantUnderflows),
	// TODO(austin) Test named types

	Val("2-1", big.NewInt(1)),
	Val("2.0-1", big.NewRat(1, 1)),
	Val("f-2", -1.0),
	Val("-0.0", big.NewRat(0, 1)),
	Val("2*2", big.NewInt(4)),
	Val("2*i", 2),
	Val("3/2", big.NewInt(1)),
	Val("3/i", 3),
	CErr("1/0", divByZero),
	CErr("1.0/0", divByZero),
	RErr("i/0", divByZero),
	Val("3%2", big.NewInt(1)),
	Val("i%2", 1),
	CErr("3%0", divByZero),
	CErr("3.0%0", opTypes),
	RErr("i%0", divByZero),

	// Examples from "Arithmetic operators"
	Val("5/3", big.NewInt(1)),
	Val("(i+4)/(i+2)", 1),
	Val("5%3", big.NewInt(2)),
	Val("(i+4)%(i+2)", 2),
	Val("-5/3", big.NewInt(-1)),
	Val("(i-6)/(i+2)", -1),
	Val("-5%3", big.NewInt(-2)),
	Val("(i-6)%(i+2)", -2),
	Val("5/-3", big.NewInt(-1)),
	Val("(i+4)/(i-4)", -1),
	Val("5%-3", big.NewInt(2)),
	Val("(i+4)%(i-4)", 2),
	Val("-5/-3", big.NewInt(1)),
	Val("(i-6)/(i-4)", 1),
	Val("-5%-3", big.NewInt(-2)),
	Val("(i-6)%(i-4)", -2),

	// Examples from "Arithmetic operators"
	Val("11/4", big.NewInt(2)),
	Val("(i+10)/4", 2),
	Val("11%4", big.NewInt(3)),
	Val("(i+10)%4", 3),
	Val("11>>2", big.NewInt(2)),
	Val("(i+10)>>2", 2),
	Val("11&3", big.NewInt(3)),
	Val("(i+10)&3", 3),
	Val("-11/4", big.NewInt(-2)),
	Val("(i-12)/4", -2),
	Val("-11%4", big.NewInt(-3)),
	Val("(i-12)%4", -3),
	Val("-11>>2", big.NewInt(-3)),
	Val("(i-12)>>2", -3),
	Val("-11&3", big.NewInt(1)),
	Val("(i-12)&3", 1),

	// TODO(austin) Test bit ops

	// For shift, we try nearly every combination of positive
	// ideal int, negative ideal int, big ideal int, ideal
	// fractional float, ideal non-fractional float, int, uint,
	// and float.
	Val("2<<2", big.NewInt(2<<2)),
	CErr("2<<(-1)", constantUnderflows),
	CErr("2<<0x10000000000000000", constantOverflows),
	CErr("2<<2.5", constantTruncated),
	Val("2<<2.0", big.NewInt(2<<2.0)),
	CErr("2<<i", mustBeUnsigned),
	Val("2<<u", 2<<1),
	CErr("2<<f", opTypes),

	Val("-2<<2", big.NewInt(-2<<2)),
	CErr("-2<<(-1)", constantUnderflows),
	CErr("-2<<0x10000000000000000", constantOverflows),
	CErr("-2<<2.5", constantTruncated),
	Val("-2<<2.0", big.NewInt(-2<<2.0)),
	CErr("-2<<i", mustBeUnsigned),
	Val("-2<<u", -2<<1),
	CErr("-2<<f", opTypes),

	Val("0x10000000000000000<<2", new(big.Int).Lsh(hugeInteger, 2)),
	CErr("0x10000000000000000<<(-1)", constantUnderflows),
	CErr("0x10000000000000000<<0x10000000000000000", constantOverflows),
	CErr("0x10000000000000000<<2.5", constantTruncated),
	Val("0x10000000000000000<<2.0", new(big.Int).Lsh(hugeInteger, 2)),
	CErr("0x10000000000000000<<i", mustBeUnsigned),
	CErr("0x10000000000000000<<u", constantOverflows),
	CErr("0x10000000000000000<<f", opTypes),

	CErr("2.5<<2", opTypes),
	CErr("2.0<<2", opTypes),

	Val("i<<2", 1<<2),
	CErr("i<<(-1)", constantUnderflows),
	CErr("i<<0x10000000000000000", constantOverflows),
	CErr("i<<2.5", constantTruncated),
	Val("i<<2.0", 1<<2),
	CErr("i<<i", mustBeUnsigned),
	Val("i<<u", 1<<1),
	CErr("i<<f", opTypes),
	Val("i<<u", 1<<1),

	Val("u<<2", uint(1<<2)),
	CErr("u<<(-1)", constantUnderflows),
	CErr("u<<0x10000000000000000", constantOverflows),
	CErr("u<<2.5", constantTruncated),
	Val("u<<2.0", uint(1<<2)),
	CErr("u<<i", mustBeUnsigned),
	Val("u<<u", uint(1<<1)),
	CErr("u<<f", opTypes),
	Val("u<<u", uint(1<<1)),

	CErr("f<<2", opTypes),

	// <, <=, >, >=
	Val("1<2", 1 < 2),
	Val("1<=2", 1 <= 2),
	Val("2<=2", 2 <= 2),
	Val("1>2", 1 > 2),
	Val("1>=2", 1 >= 2),
	Val("2>=2", 2 >= 2),

	Val("i<2", 1 < 2),
	Val("i<=2", 1 <= 2),
	Val("i+1<=2", 2 <= 2),
	Val("i>2", 1 > 2),
	Val("i>=2", 1 >= 2),
	Val("i+1>=2", 2 >= 2),

	Val("u<2", 1 < 2),
	Val("f<2", 1 < 2),

	Val("s<\"b\"", true),
	Val("s<\"a\"", false),
	Val("s<=\"abc\"", true),
	Val("s>\"aa\"", true),
	Val("s>\"ac\"", false),
	Val("s>=\"abc\"", true),

	CErr("i<u", opTypes),
	CErr("i<f", opTypes),
	CErr("i<s", opTypes),
	CErr("&i<&i", opTypes),
	CErr("ai<ai", opTypes),

	// ==, !=
	Val("1==1", true),
	Val("1!=1", false),
	Val("1==2", false),
	Val("1!=2", true),

	Val("1.0==1", true),
	Val("1.5==1", false),

	Val("i==1", true),
	Val("i!=1", false),
	Val("i==2", false),
	Val("i!=2", true),

	Val("u==1", true),
	Val("f==1", true),

	Val("s==\"abc\"", true),
	Val("s!=\"abc\"", false),
	Val("s==\"abcd\"", false),
	Val("s!=\"abcd\"", true),

	Val("&i==&i", true),
	Val("&i==&i2", false),

	Val("fn==fn", true),
	Val("fn==func(int)int{return 0}", false),

	CErr("i==u", opTypes),
	CErr("i==f", opTypes),
	CErr("&i==&f", opTypes),
	CErr("ai==ai", opTypes),
	CErr("t==t", opTypes),
	CErr("fn==oneTwo", opTypes),
}

func TestExpr(t *testing.T) { runTests(t, "exprTests", exprTests) }
