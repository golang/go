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
type ES struct{}

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
