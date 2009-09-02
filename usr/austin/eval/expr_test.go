// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eval

import (
	"bignum";
	"testing";
)

var undefined = "undefined"
var typeAsExpr = "type .* used as expression"
var badCharLit = "character literal"
var illegalEscape = "illegal char escape"
var opTypes = "illegal (operand|argument) type|cannot index into"
var badAddrOf = "cannot take the address"
var constantTruncated = "constant [^ ]* truncated"
var constantUnderflows = "constant [^ ]* underflows"
var constantOverflows = "constant [^ ]* overflows"
var implLimit = "implementation limit"
var mustBeUnsigned = "must be unsigned"
var divByZero = "divide by zero"

var hugeInteger = bignum.Int(1).Shl(64);

var exprTests = []test {
	Val("i", 1),
	EErr("zzz", undefined),
	// TODO(austin) Test variable in constant context
	//EErr("t", typeAsExpr),

	Val("'a'", bignum.Int('a')),
	Val("'\\uffff'", bignum.Int('\uffff')),
	Val("'\\n'", bignum.Int('\n')),
	EErr("''+x", badCharLit),
	// Produces two parse errors
	//EErr("'''", ""),
	EErr("'\n'", badCharLit),
	EErr("'\\z'", illegalEscape),
	EErr("'ab'", badCharLit),

	Val("1.0", bignum.Rat(1, 1)),
	Val("1.", bignum.Rat(1, 1)),
	Val(".1", bignum.Rat(1, 10)),
	Val("1e2", bignum.Rat(100, 1)),

	Val("\"abc\"", "abc"),
	Val("\"\"", ""),
	Val("\"\\n\\\"\"", "\n\""),
	EErr("\"\\z\"", illegalEscape),
	EErr("\"abc", "string not terminated"),

	Val("\"abc\" \"def\"", "abcdef"),
	EErr("\"abc\" \"\\z\"", illegalEscape),

	Val("(i)", 1),

	Val("ai[0]", 1),
	Val("(&ai)[0]", 1),
	Val("ai[1]", 2),
	Val("ai[i]", 2),
	Val("ai[u]", 2),
	EErr("ai[f]", opTypes),
	EErr("ai[0][0]", opTypes),
	EErr("ai[2]", "index 2 exceeds"),
	EErr("ai[1+1]", "index 2 exceeds"),
	EErr("ai[-1]", "negative index"),
	ERTErr("ai[i+i]", "index 2 exceeds"),
	ERTErr("ai[-i]", "negative index"),
	EErr("i[0]", opTypes),
	EErr("f[0]", opTypes),

	Val("aai[0][0]", 1),
	Val("aai[1][1]", 4),
	EErr("aai[2][0]", "index 2 exceeds"),
	EErr("aai[0][2]", "index 2 exceeds"),

	Val("sli[0]", 1),
	Val("sli[1]", 2),
	EErr("sli[-1]", "negative index"),
	ERTErr("sli[-i]", "negative index"),
	ERTErr("sli[2]", "index 2 exceeds"),

	Val("s[0]", uint8('a')),
	Val("s[1]", uint8('b')),
	EErr("s[-1]", "negative index"),
	ERTErr("s[-i]", "negative index"),
	ERTErr("s[3]", "index 3 exceeds"),

	EErr("1(2)", "cannot call"),
	EErr("fn(1,2)", "too many"),
	EErr("fn()", "not enough"),
	EErr("fn(true)", opTypes),
	EErr("fn(true)", "function call"),
	// Single argument functions don't say which argument.
	//EErr("fn(true)", "argument 1"),
	Val("fn(1)", 2),
	Val("fn(1.0)", 2),
	EErr("fn(1.5)", constantTruncated),
	Val("fn(i)", 2),
	EErr("fn(u)", opTypes),

	EErr("void()+2", opTypes),
	EErr("oneTwo()+2", opTypes),

	Val("cap(ai)", 2),
	Val("cap(&ai)", 2),
	Val("cap(aai)", 2),
	Val("cap(sli)", 3),
	EErr("cap(0)", opTypes),
	EErr("cap(i)", opTypes),
	EErr("cap(s)", opTypes),

	Val("len(s)", 3),
	Val("len(ai)", 2),
	Val("len(&ai)", 2),
	Val("len(aai)", 2),
	Val("len(sli)", 2),
	// TODO(austin) Test len of map
	EErr("len(0)", opTypes),
	EErr("len(i)", opTypes),

	EErr("*i", opTypes),
	Val("*&i", 1),
	Val("*&(i)", 1),
	EErr("&1", badAddrOf),
	EErr("&c", badAddrOf),
	Val("*(&ai[0])", 1),

	Val("+1", bignum.Int(+1)),
	Val("+1.0", bignum.Rat(1, 1)),
	EErr("+\"x\"", opTypes),

	Val("-42", bignum.Int(-42)),
	Val("-i", -1),
	Val("-f", -1.0),
	// 6g bug?
	//Val("-(f-1)", -0.0),
	EErr("-\"x\"", opTypes),

	// TODO(austin) Test unary !

	Val("^2", bignum.Int(^2)),
	Val("^(-2)", bignum.Int(^(-2))),
	EErr("^2.0", opTypes),
	EErr("^2.5", opTypes),
	Val("^i", ^1),
	Val("^u", ^uint(1)),
	EErr("^f", opTypes),

	Val("1+i", 2),
	Val("1+u", uint(2)),
	Val("3.0+i", 4),
	Val("1+1", bignum.Int(2)),
	Val("f+f", 2.0),
	Val("1+f", 2.0),
	Val("1.0+1", bignum.Rat(2, 1)),
	Val("\"abc\" + \"def\"", "abcdef"),
	EErr("i+u", opTypes),
	EErr("-1+u", constantUnderflows),
	// TODO(austin) Test named types

	Val("2-1", bignum.Int(1)),
	Val("2.0-1", bignum.Rat(1, 1)),
	Val("f-2", -1.0),
	// TOOD(austin) bignum can't do negative 0?
	//Val("-0.0", XXX),
	Val("2*2", bignum.Int(4)),
	Val("2*i", 2),
	Val("3/2", bignum.Int(1)),
	Val("3/i", 3),
	EErr("1/0", divByZero),
	EErr("1.0/0", divByZero),
	ERTErr("i/0", divByZero),
	Val("3%2", bignum.Int(1)),
	Val("i%2", 1),
	EErr("3%0", divByZero),
	EErr("3.0%0", opTypes),
	ERTErr("i%0", divByZero),

	// Examples from "Arithmetic operators"
	Val("5/3", bignum.Int(1)),
	Val("(i+4)/(i+2)", 1),
	Val("5%3", bignum.Int(2)),
	Val("(i+4)%(i+2)", 2),
	Val("-5/3", bignum.Int(-1)),
	Val("(i-6)/(i+2)", -1),
	Val("-5%3", bignum.Int(-2)),
	Val("(i-6)%(i+2)", -2),
	Val("5/-3", bignum.Int(-1)),
	Val("(i+4)/(i-4)", -1),
	Val("5%-3", bignum.Int(2)),
	Val("(i+4)%(i-4)", 2),
	Val("-5/-3", bignum.Int(1)),
	Val("(i-6)/(i-4)", 1),
	Val("-5%-3", bignum.Int(-2)),
	Val("(i-6)%(i-4)", -2),

	// Examples from "Arithmetic operators"
	Val("11/4", bignum.Int(2)),
	Val("(i+10)/4", 2),
	Val("11%4", bignum.Int(3)),
	Val("(i+10)%4", 3),
	Val("11>>2", bignum.Int(2)),
	Val("(i+10)>>2", 2),
	Val("11&3", bignum.Int(3)),
	Val("(i+10)&3", 3),
	Val("-11/4", bignum.Int(-2)),
	Val("(i-12)/4", -2),
	Val("-11%4", bignum.Int(-3)),
	Val("(i-12)%4", -3),
	Val("-11>>2", bignum.Int(-3)),
	Val("(i-12)>>2", -3),
	Val("-11&3", bignum.Int(1)),
	Val("(i-12)&3", 1),

	// TODO(austin) Test bit ops

	// For shift, we try nearly every combination of positive
	// ideal int, negative ideal int, big ideal int, ideal
	// fractional float, ideal non-fractional float, int, uint,
	// and float.
	Val("2<<2", bignum.Int(2<<2)),
	EErr("2<<(-1)", constantUnderflows),
	EErr("2<<0x10000000000000000", constantOverflows),
	EErr("2<<2.5", constantTruncated),
	Val("2<<2.0", bignum.Int(2<<2.0)),
	EErr("2<<i", mustBeUnsigned),
	Val("2<<u", 2<<1),
	EErr("2<<f", opTypes),

	Val("-2<<2", bignum.Int(-2<<2)),
	EErr("-2<<(-1)", constantUnderflows),
	EErr("-2<<0x10000000000000000", constantOverflows),
	EErr("-2<<2.5", constantTruncated),
	Val("-2<<2.0", bignum.Int(-2<<2.0)),
	EErr("-2<<i", mustBeUnsigned),
	Val("-2<<u", -2<<1),
	EErr("-2<<f", opTypes),

	Val("0x10000000000000000<<2", hugeInteger.Shl(2)),
	EErr("0x10000000000000000<<(-1)", constantUnderflows),
	EErr("0x10000000000000000<<0x10000000000000000", constantOverflows),
	EErr("0x10000000000000000<<2.5", constantTruncated),
	Val("0x10000000000000000<<2.0", hugeInteger.Shl(2)),
	EErr("0x10000000000000000<<i", mustBeUnsigned),
	EErr("0x10000000000000000<<u", constantOverflows),
	EErr("0x10000000000000000<<f", opTypes),

	EErr("2.5<<2", opTypes),
	EErr("2.0<<2", opTypes),

	Val("i<<2", 1<<2),
	EErr("i<<(-1)", constantUnderflows),
	EErr("i<<0x10000000000000000", constantOverflows),
	EErr("i<<2.5", constantTruncated),
	Val("i<<2.0", 1<<2),
	EErr("i<<i", mustBeUnsigned),
	Val("i<<u", 1<<1),
	EErr("i<<f", opTypes),
	Val("i<<u", 1<<1),

	Val("u<<2", uint(1<<2)),
	EErr("u<<(-1)", constantUnderflows),
	EErr("u<<0x10000000000000000", constantOverflows),
	EErr("u<<2.5", constantTruncated),
	Val("u<<2.0", uint(1<<2)),
	EErr("u<<i", mustBeUnsigned),
	Val("u<<u", uint(1<<1)),
	EErr("u<<f", opTypes),
	Val("u<<u", uint(1<<1)),

	EErr("f<<2", opTypes),

	// <, <=, >, >=
	Val("1<2", 1<2),
	Val("1<=2", 1<=2),
	Val("2<=2", 2<=2),
	Val("1>2", 1>2),
	Val("1>=2", 1>=2),
	Val("2>=2", 2>=2),

	Val("i<2", 1<2),
	Val("i<=2", 1<=2),
	Val("i+1<=2", 2<=2),
	Val("i>2", 1>2),
	Val("i>=2", 1>=2),
	Val("i+1>=2", 2>=2),

	Val("u<2", 1<2),
	Val("f<2", 1<2),

	Val("s<\"b\"", true),
	Val("s<\"a\"", false),
	Val("s<=\"abc\"", true),
	Val("s>\"aa\"", true),
	Val("s>\"ac\"", false),
	Val("s>=\"abc\"", true),

	EErr("i<u", opTypes),
	EErr("i<f", opTypes),
	EErr("i<s", opTypes),
	EErr("&i<&i", opTypes),
	EErr("ai<ai", opTypes),

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

	EErr("i==u", opTypes),
	EErr("i==f", opTypes),
	EErr("&i==&f", opTypes),
	EErr("ai==ai", opTypes),
	EErr("t==t", opTypes),
	EErr("fn==oneTwo", opTypes),
}

func TestExpr(t *testing.T) {
	runTests(t, "exprTests", exprTests);
}
