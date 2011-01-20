// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package datafmt

import (
	"fmt"
	"testing"
	"go/token"
)


var fset = token.NewFileSet()


func parse(t *testing.T, form string, fmap FormatterMap) Format {
	f, err := Parse(fset, "", []byte(form), fmap)
	if err != nil {
		t.Errorf("Parse(%s): %v", form, err)
		return nil
	}
	return f
}


func verify(t *testing.T, f Format, expected string, args ...interface{}) {
	if f == nil {
		return // allow other tests to run
	}
	result := f.Sprint(args...)
	if result != expected {
		t.Errorf(
			"result  : `%s`\nexpected: `%s`\n\n",
			result, expected)
	}
}


func formatter(s *State, value interface{}, rule_name string) bool {
	switch rule_name {
	case "/":
		fmt.Fprintf(s, "%d %d %d", s.Pos().Line, s.LinePos().Column, s.Pos().Column)
		return true
	case "blank":
		s.Write([]byte{' '})
		return true
	case "int":
		if value.(int)&1 == 0 {
			fmt.Fprint(s, "even ")
		} else {
			fmt.Fprint(s, "odd ")
		}
		return true
	case "nil":
		return false
	case "testing.T":
		s.Write([]byte("testing.T"))
		return true
	}
	panic("unreachable")
	return false
}


func TestCustomFormatters(t *testing.T) {
	fmap0 := FormatterMap{"/": formatter}
	fmap1 := FormatterMap{"int": formatter, "blank": formatter, "nil": formatter}
	fmap2 := FormatterMap{"testing.T": formatter}

	f := parse(t, `int=`, fmap0)
	verify(t, f, ``, 1, 2, 3)

	f = parse(t, `int="#"`, nil)
	verify(t, f, `###`, 1, 2, 3)

	f = parse(t, `int="#";string="%s"`, fmap0)
	verify(t, f, "#1 0 1#1 0 7#1 0 13\n2 0 0foo2 0 8\n", 1, 2, 3, "\n", "foo", "\n")

	f = parse(t, ``, fmap1)
	verify(t, f, `even odd even odd `, 0, 1, 2, 3)

	f = parse(t, `/ =@:blank; float64="#"`, fmap1)
	verify(t, f, `# # #`, 0.0, 1.0, 2.0)

	f = parse(t, `float64=@:nil`, fmap1)
	verify(t, f, ``, 0.0, 1.0, 2.0)

	f = parse(t, `testing "testing"; ptr=*`, fmap2)
	verify(t, f, `testing.T`, t)

	// TODO needs more tests
}


// ----------------------------------------------------------------------------
// Formatting of basic and simple composite types

func check(t *testing.T, form, expected string, args ...interface{}) {
	f := parse(t, form, nil)
	if f == nil {
		return // allow other tests to run
	}
	result := f.Sprint(args...)
	if result != expected {
		t.Errorf(
			"format  : %s\nresult  : `%s`\nexpected: `%s`\n\n",
			form, result, expected)
	}
}


func TestBasicTypes(t *testing.T) {
	check(t, ``, ``)
	check(t, `bool=":%v"`, `:true:false`, true, false)
	check(t, `int="%b %d %o 0x%x"`, `101010 42 52 0x2a`, 42)

	check(t, `int="%"`, `%`, 42)
	check(t, `int="%%"`, `%`, 42)
	check(t, `int="**%%**"`, `**%**`, 42)
	check(t, `int="%%%%%%"`, `%%%`, 42)
	check(t, `int="%%%d%%"`, `%42%`, 42)

	const i = -42
	const is = `-42`
	check(t, `int  ="%d"`, is, i)
	check(t, `int8 ="%d"`, is, int8(i))
	check(t, `int16="%d"`, is, int16(i))
	check(t, `int32="%d"`, is, int32(i))
	check(t, `int64="%d"`, is, int64(i))

	const u = 42
	const us = `42`
	check(t, `uint  ="%d"`, us, uint(u))
	check(t, `uint8 ="%d"`, us, uint8(u))
	check(t, `uint16="%d"`, us, uint16(u))
	check(t, `uint32="%d"`, us, uint32(u))
	check(t, `uint64="%d"`, us, uint64(u))

	const f = 3.141592
	const fs = `3.141592`
	check(t, `float64="%g"`, fs, f)
	check(t, `float32="%g"`, fs, float32(f))
	check(t, `float64="%g"`, fs, float64(f))
}


func TestArrayTypes(t *testing.T) {
	var a0 [10]int
	check(t, `array="array";`, `array`, a0)

	a1 := [...]int{1, 2, 3}
	check(t, `array="array";`, `array`, a1)
	check(t, `array={*}; int="%d";`, `123`, a1)
	check(t, `array={* / ", "}; int="%d";`, `1, 2, 3`, a1)
	check(t, `array={* / *}; int="%d";`, `12233`, a1)

	a2 := []interface{}{42, "foo", 3.14}
	check(t, `array={* / ", "}; interface=*; string="bar"; default="%v";`, `42, bar, 3.14`, a2)
}


func TestChanTypes(t *testing.T) {
	var c0 chan int
	check(t, `chan="chan"`, `chan`, c0)

	c1 := make(chan int)
	go func() { c1 <- 42 }()
	check(t, `chan="chan"`, `chan`, c1)
	// check(t, `chan=*`, `42`, c1);  // reflection support for chans incomplete
}


func TestFuncTypes(t *testing.T) {
	var f0 func() int
	check(t, `func="func"`, `func`, f0)

	f1 := func() int { return 42 }
	check(t, `func="func"`, `func`, f1)
	// check(t, `func=*`, `42`, f1);  // reflection support for funcs incomplete
}


func TestMapTypes(t *testing.T) {
	var m0 map[string]int
	check(t, `map="map"`, `map`, m0)

	m1 := map[string]int{}
	check(t, `map="map"`, `map`, m1)
	// check(t, `map=*`, ``, m1);  // reflection support for maps incomplete
}


func TestPointerTypes(t *testing.T) {
	var p0 *int
	check(t, `ptr="ptr"`, `ptr`, p0)
	check(t, `ptr=*`, ``, p0)
	check(t, `ptr=*|"nil"`, `nil`, p0)

	x := 99991
	p1 := &x
	check(t, `ptr="ptr"`, `ptr`, p1)
	check(t, `ptr=*; int="%d"`, `99991`, p1)
}


func TestDefaultRule(t *testing.T) {
	check(t, `default="%v"`, `42foo3.14`, 42, "foo", 3.14)
	check(t, `default="%v"; int="%x"`, `abcdef`, 10, 11, 12, 13, 14, 15)
	check(t, `default="%v"; int="%x"`, `ab**ef`, 10, 11, "**", 14, 15)
	check(t, `default="%x"; int=@:default`, `abcdef`, 10, 11, 12, 13, 14, 15)
}


func TestGlobalSeparatorRule(t *testing.T) {
	check(t, `int="%d"; / ="-"`, `1-2-3-4`, 1, 2, 3, 4)
	check(t, `int="%x%x"; / ="*"`, `aa*aa`, 10, 10)
}


// ----------------------------------------------------------------------------
// Formatting of a struct

type T1 struct {
	a int
}

const F1 = `datafmt "datafmt";` +
	`int = "%d";` +
	`datafmt.T1 = "<" a ">";`

func TestStruct1(t *testing.T) { check(t, F1, "<42>", T1{42}) }


// ----------------------------------------------------------------------------
// Formatting of a struct with an optional field (ptr)

type T2 struct {
	s string
	p *T1
}

const F2a = F1 +
	`string = "%s";` +
	`ptr = *;` +
	`datafmt.T2 = s ["-" p "-"];`

const F2b = F1 +
	`string = "%s";` +
	`ptr = *;` +
	`datafmt.T2 = s ("-" p "-" | "empty");`

func TestStruct2(t *testing.T) {
	check(t, F2a, "foo", T2{"foo", nil})
	check(t, F2a, "bar-<17>-", T2{"bar", &T1{17}})
	check(t, F2b, "fooempty", T2{"foo", nil})
}


// ----------------------------------------------------------------------------
// Formatting of a struct with a repetitive field (slice)

type T3 struct {
	s string
	a []int
}

const F3a = `datafmt "datafmt";` +
	`default = "%v";` +
	`array = *;` +
	`datafmt.T3 = s  {" " a a / ","};`

const F3b = `datafmt "datafmt";` +
	`int = "%d";` +
	`string = "%s";` +
	`array = *;` +
	`nil = ;` +
	`empty = *:nil;` +
	`datafmt.T3 = s [a:empty ": " {a / "-"}]`

func TestStruct3(t *testing.T) {
	check(t, F3a, "foo", T3{"foo", nil})
	check(t, F3a, "foo 00, 11, 22", T3{"foo", []int{0, 1, 2}})
	check(t, F3b, "bar", T3{"bar", nil})
	check(t, F3b, "bal: 2-3-5", T3{"bal", []int{2, 3, 5}})
}


// ----------------------------------------------------------------------------
// Formatting of a struct with alternative field

type T4 struct {
	x *int
	a []int
}

const F4a = `datafmt "datafmt";` +
	`int = "%d";` +
	`ptr = *;` +
	`array = *;` +
	`nil = ;` +
	`empty = *:nil;` +
	`datafmt.T4 = "<" (x:empty x | "-") ">" `

const F4b = `datafmt "datafmt";` +
	`int = "%d";` +
	`ptr = *;` +
	`array = *;` +
	`nil = ;` +
	`empty = *:nil;` +
	`datafmt.T4 = "<" (a:empty {a / ", "} | "-") ">" `

func TestStruct4(t *testing.T) {
	x := 7
	check(t, F4a, "<->", T4{nil, nil})
	check(t, F4a, "<7>", T4{&x, nil})
	check(t, F4b, "<->", T4{nil, nil})
	check(t, F4b, "<2, 3, 7>", T4{nil, []int{2, 3, 7}})
}


// ----------------------------------------------------------------------------
// Formatting a struct (documentation example)

type Point struct {
	name string
	x, y int
}

const FPoint = `datafmt "datafmt";` +
	`int = "%d";` +
	`hexInt = "0x%x";` +
	`string = "---%s---";` +
	`datafmt.Point = name "{" x ", " y:hexInt "}";`

func TestStructPoint(t *testing.T) {
	p := Point{"foo", 3, 15}
	check(t, FPoint, "---foo---{3, 0xf}", p)
}


// ----------------------------------------------------------------------------
// Formatting a slice (documentation example)

const FSlice = `int = "%b";` +
	`array = { * / ", " }`

func TestSlice(t *testing.T) { check(t, FSlice, "10, 11, 101, 111", []int{2, 3, 5, 7}) }


// TODO add more tests
