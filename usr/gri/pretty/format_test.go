// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package format

import (
	"format";
	"testing";
)


func check(t *testing.T, form, expected string, args ...) {
	result := format.Parse(form).Sprint(args);
	if result != expected {
		t.Errorf(
			"format  : %s\nresult  : %s\nexpected: %s\n\n",
			form, result, expected
		)
	}
}


// ----------------------------------------------------------------------------
// - formatting of basic type int

const F0 =
	`int = "0x%x";`

func Test0(t *testing.T) {
	check(t, F0, "0x2a", 42);
}


// ----------------------------------------------------------------------------
// - default formatting of basic type int
// - formatting of a struct

type T1 struct {
	a int;
}

const F1 =
	`format.T1 = "<" a ">";`

func Test1(t *testing.T) {
	check(t, F1, "<42>", T1{42});
}


// ----------------------------------------------------------------------------
// - formatting of a struct with an optional field (pointer)
// - default formatting for pointers

type T2 struct {
	s string;
	p *T1;
}

const F2a =
	F1 +
	`pointer = *;`
	`format.T2 = s ["-" p "-"];`;
	
const F2b =
	F1 +
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
	`format.T3 = s  { " " a a "," };`;

const F3b =
	`format.T3 = [a:""] s | "nothing";`;  // use 'a' to select alternative w/o printing a

func Test3(t *testing.T) {
	check(t, F3a, "foo", T3{"foo", nil});
	check(t, F3a, "foo 00, 11, 22,", T3{"foo", []int{0, 1, 2}});
	//check(t, F3b, "nothing", T3{"bar", nil});  // TODO fix this
	check(t, F3b, "bar", T3{"bar", []int{0}});
}
