// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package format

import (
	"format";
	"io";
	"testing";
)


func check(t *testing.T, form, expected string, args ...) {
	f, err := format.Parse(io.StringBytes(form), nil);
	if err != nil {
		panic(form + ": " + err.String());
	}
	result := f.Sprint(args);
	if result != expected {
		t.Errorf(
			"format  : %s\nresult  : `%s`\nexpected: `%s`\n\n",
			form, result, expected
		)
	}
}


// ----------------------------------------------------------------------------
// Syntax

func TestA(t *testing.T) {
	// TODO fill this in
}


// ----------------------------------------------------------------------------
// - formatting of basic types

func Test0(t *testing.T) {
	check(t, `bool = "%v"`, "false", false);
	check(t, `int = "%b %d %o 0x%x"`, "101010 42 52 0x2a", 42);
}


// ----------------------------------------------------------------------------
// - formatting of a struct

type T1 struct {
	a int;
}

const F1 =
	`format "format";`
	`int = "%d";`
	`format.T1 = "<" a ">";`

func Test1(t *testing.T) {
	check(t, F1, "<42>", T1{42});
}


// ----------------------------------------------------------------------------
// - formatting of a struct with an optional field (pointer)

type T2 struct {
	s string;
	p *T1;
}

const F2a =
	F1 +
	`string = "%s";`
	`pointer = *;`
	`format.T2 = s ["-" p "-"];`
	
const F2b =
	F1 +
	`string = "%s";`
	`pointer = *;`
	`format.T2 = s ("-" p "-" | "empty");`;
	
func Test2(t *testing.T) {
	check(t, F2a, "foo", T2{"foo", nil});
	check(t, F2a, "bar-<17>-", T2{"bar", &T1{17}});
	check(t, F2b, "fooempty", T2{"foo", nil});
}


// ----------------------------------------------------------------------------
// - formatting of a struct with a repetitive field (slice)

type T3 struct {
	s string;
	a []int;
}

const F3a =
	`format "format";`
	`default = "%v";`
	`array = *;`
	`format.T3 = s  {" " a a / ","};`

const F3b =
	`format "format";`
	`int = "%d";`
	`string = "%s";`
	`array = *;`
	`nil = ;`
	`empty = *:nil;`
	`format.T3 = s [a:empty ": " {a / "-"}]`

func Test3(t *testing.T) {
	check(t, F3a, "foo", T3{"foo", nil});
	check(t, F3a, "foo 00, 11, 22", T3{"foo", []int{0, 1, 2}});
	check(t, F3b, "bar", T3{"bar", nil});
	check(t, F3b, "bal: 2-3-5", T3{"bal", []int{2, 3, 5}});
}


// ----------------------------------------------------------------------------
// - formatting of a struct with alternative field

type T4 struct {
	x *int;
	a []int;
}

const F4a =
	`format "format";`
	`int = "%d";`
	`pointer = *;`
	`array = *;`
	`nil = ;`
	`empty = *:nil;`
	`format.T4 = "<" (x:empty x | "-") ">" `

const F4b =
	`format "format";`
	`int = "%d";`
	`pointer = *;`
	`array = *;`
	`nil = ;`
	`empty = *:nil;`
	`format.T4 = "<" (a:empty {a / ", "} | "-") ">" `

func Test4(t *testing.T) {
	x := 7;
	check(t, F4a, "<->", T4{nil, nil});
	check(t, F4a, "<7>", T4{&x, nil});
	check(t, F4b, "<->", T4{nil, nil});
	check(t, F4b, "<2, 3, 7>", T4{nil, []int{2, 3, 7}});
}
