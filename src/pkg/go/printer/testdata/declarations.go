// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package imports

import "io"

import (
	_ "io"
)

import _ "io"

import (
	"io";
	"io";
	"io";
)

import (
	"io";
	aLongRename "io";
	b "io";
	c "i" "o";
)

// no newlines between consecutive single imports, but
// respect extra line breaks in the source (at most one empty line)
import _ "io"
import _ "io"
import _ "io"

import _ "os"
import _ "os"
import _ "os"


import _ "fmt"
import _ "fmt"
import _ "fmt"


// at least one empty line between declarations of different kind
import _ "io"
var _ int;


func _() {
	// the following decls need a semicolon at the end
	type _ int;
	type _ *int;
	type _ []int;
	type _ map[string]int;
	type _ chan int;
	type _ func() int;

	var _ int;
	var _ *int;
	var _ []int;
	var _ map[string]int;
	var _ chan int;
	var _ func() int;

	// the following decls don't need a semicolon at the end
	type _ struct{}
	type _ *struct{}
	type _ []struct{}
	type _ map[string]struct{}
	type _ chan struct{}
	type _ func() struct{}

	type _ interface{}
	type _ *interface{}
	type _ []interface{}
	type _ map[string]interface{}
	type _ chan interface{}
	type _ func() interface{}

	var _ struct{}
	var _ *struct{}
	var _ []struct{}
	var _ map[string]struct{}
	var _ chan struct{}
	var _ func() struct{}

	var _ interface{}
	var _ *interface{}
	var _ []interface{}
	var _ map[string]interface{}
	var _ chan interface{}
	var _ func() interface{}
}




// no tabs for single or ungrouped decls
func _() {
	const xxxxxx = 0;
	type x int;
	var xxx int;
	var yyyy float = 3.14;
	var zzzzz = "bar";

	const (
		xxxxxx = 0;
	)
	type (
		x int;
	)
	var (
		xxx int;
	)
	var (
		yyyy float = 3.14;
	)
	var (
		zzzzz = "bar";
	)
}

// tabs for multiple or grouped decls
func _() {
	// no entry has a type
	const (
		zzzzzz = 1;
		z = 2;
		zzz = 3;
	)
	// some entries have a type
	const (
		xxxxxx = 1;
		x = 2;
		xxx = 3;
		yyyyyyyy float = iota;
		yyyy = "bar";
		yyy;
		yy = 2;
	)
}

func _() {
	// no entry has a type
	var (
		zzzzzz = 1;
		z = 2;
		zzz = 3;
	)
	// no entry has a value
	var (
		_ int;
		_ float;
		_ string;

		_ int;  // comment
		_ float;  // comment
		_ string;  // comment
	)
	// some entries have a type
	var (
		xxxxxx int;
		x float;
		xxx string;
		yyyyyyyy int = 1234;
		y float = 3.14;
		yyyy = "bar";
		yyy string = "foo";
	)
	// mixed entries - all comments should be aligned
	var (
		a, b, c int;
		x = 10;
		d int;  // comment
		y = 20;  // comment
		f, ff, fff, ffff int = 0, 1, 2, 3;  // comment
	)
	// respect original line breaks
	var _ = []T {
		T{0x20,	"Telugu"}
	};
	var _ = []T {
		// respect original line breaks
		T{0x20,	"Telugu"}
	};
}

func _() {
	type (
		xxxxxx int;
		x float;
		xxx string;
		xxxxx []x;
		xx struct{};
		xxxxxxx struct {
			_, _ int;
			_ float;
		};
		xxxx chan<- string;
	)
}


// formatting of structs
type _ struct{}

type _ struct{ /* this comment should be visible */ }

type _ struct{
	// this comment should be visible and properly indented
}

type _ struct {  // this comment must not change indentation
	f int;
	f, ff, fff, ffff int;
}

type _ struct {
	string;
}

type _ struct {
	string;  // comment
}

type _ struct {
	string "tag"
}

type _ struct {
	string "tag"  // comment
}

type _ struct {
	f int;
}

type _ struct {
	f int;  // comment
}

type _ struct {
	f int "tag";
}

type _ struct {
	f int "tag";  // comment
}

type _ struct {
	bool;
	a, b, c int;
	int "tag";
	ES; // comment
	float "tag";  // comment
	f int;  // comment
	f, ff, fff, ffff int;  // comment
	g float "tag";
	h float "tag";  // comment
}


// difficult cases
type _ struct {
	bool;  // comment
	text []byte;  // comment
}



// formatting of interfaces
type EI interface{}

type _ interface {
	EI;
}

type _ interface {
	f();
	fffff();
}

type _ interface {
	EI;
	f();
	fffffg();
}

type _ interface {  // this comment must not change indentation
	EI;  // here's a comment
	f();  // no blank between identifier and ()
	fffff();  // no blank between identifier and ()
	gggggggggggg(x, y, z int) ();  // hurray
}

// formatting of variable declarations
func _() {
	type day struct { n int; short, long string };
	var (
		Sunday = day{ 0, "SUN", "Sunday" };
		Monday = day{ 1, "MON", "Monday" };
		Tuesday = day{ 2, "TUE", "Tuesday" };
		Wednesday = day{ 3, "WED", "Wednesday" };
		Thursday = day{ 4, "THU", "Thursday" };
		Friday = day{ 5, "FRI", "Friday" };
		Saturday = day{ 6, "SAT", "Saturday" };
	)
}


func _() {
	var Universe = Scope {
		Names: map[string]*Ident {
			// basic types
			"bool": nil,
			"byte": nil,
			"int8": nil,
			"int16": nil,
			"int32": nil,
			"int64": nil,
			"uint8": nil,
			"uint16": nil,
			"uint32": nil,
			"uint64": nil,
			"float32": nil,
			"float64": nil,
			"string": nil,

			// convenience types
			"int": nil,
			"uint": nil,
			"uintptr": nil,
			"float": nil,

			// constants
			"false": nil,
			"true": nil,
			"iota": nil,
			"nil": nil,

			// functions
			"cap": nil,
			"len": nil,
			"new": nil,
			"make": nil,
			"panic": nil,
			"panicln": nil,
			"print": nil,
			"println": nil,
		}
	}
}


// formatting of consecutive single-line functions
func _() {}
func _() {}
func _() {}

func _() {}  // an empty line before this function
func _() {}
func _() {}
