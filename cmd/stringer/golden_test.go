// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains simple golden tests for various examples.
// Besides validating the results when the implementation changes,
// it provides a way to look at the generated code without having
// to execute the print statements in one's head.

package main

import (
	"strings"
	"testing"
)

// Golden represents a test case.
type Golden struct {
	name   string
	input  string // input; the package clause is provided when running the test.
	output string // exected output.
}

var golden = []Golden{
	{"day", day_in, day_out},
	{"offset", offset_in, offset_out},
	{"gap", gap_in, gap_out},
	{"neg", neg_in, neg_out},
	{"uneg", uneg_in, uneg_out},
	{"map", map_in, map_out},
}

// Each example starts with "type XXX [u]int", with a single space separating them.

// Simple test: enumeration of type int starting at 0.
const day_in = `type Day int
const (
	Monday Day = iota
	Tuesday
	Wednesday
	Thursday
	Friday
	Saturday
	Sunday
)
`

const day_out = `
var (
	_Day_indexes = [...]uint8{6, 13, 22, 30, 36, 44, 50}
	_Day_names   = "MondayTuesdayWednesdayThursdayFridaySaturdaySunday"
)

func (i Day) String() string {
	if i < 0 || i >= Day(len(_Day_indexes)) {
		return fmt.Sprintf("Day(%d)", i)
	}
	hi := _Day_indexes[i]
	lo := uint8(0)
	if i > 0 {
		lo = _Day_indexes[i-1]
	}
	return _Day_names[lo:hi]
}
`

// Enumeration with an offset.
// Also includes a duplicate.
const offset_in = `type Number int
const (
	_ Number = iota
	One
	Two
	Three
	AnotherOne = One  // Duplicate; note that AnotherOne doesn't appear below.
)
`

const offset_out = `
var (
	_Number_indexes = [...]uint8{3, 6, 11}
	_Number_names   = "OneTwoThree"
)

func (i Number) String() string {
	i -= 1
	if i < 0 || i >= Number(len(_Number_indexes)) {
		return fmt.Sprintf("Number(%d)", i+1)
	}
	hi := _Number_indexes[i]
	lo := uint8(0)
	if i > 0 {
		lo = _Number_indexes[i-1]
	}
	return _Number_names[lo:hi]
}
`

// Gaps and an offset.
const gap_in = `type Num int
const (
	Two Num = 2
	Three Num = 3
	Five Num = 5
	Six Num = 6
	Seven Num = 7
	Eight Num = 8
	Nine Num = 9
	Eleven Num = 11
)
`

const gap_out = `
var (
	_Num_indexes_0 = [...]uint8{3, 8}
	_Num_names_0   = "TwoThree"
	_Num_indexes_1 = [...]uint8{4, 7, 12, 17, 21}
	_Num_names_1   = "FiveSixSevenEightNine"
	_Num_indexes_2 = [...]uint8{6}
	_Num_names_2   = "Eleven"
)

func (i Num) String() string {
	switch {
	case 2 <= i && i < 3:
		lo := uint8(0)
		if i > 2 {
			i -= 2
		} else {
			lo = _Num_indexes_0[i-1]
		}
		return _Num_names_0[lo:_Num_indexes_0[i]]
	case 5 <= i && i < 9:
		lo := uint8(0)
		if i > 5 {
			i -= 5
		} else {
			lo = _Num_indexes_1[i-1]
		}
		return _Num_names_1[lo:_Num_indexes_1[i]]
	case i == 11:
		return _Num_names_2
	default:
		return fmt.Sprintf("Num(%d)", i)
	}
}
`

// Signed integers spanning zero.
const neg_in = `type Num int
const (
	m_2 Num = -2 + iota
	m_1
	m0
	m1
	m2
)
`

const neg_out = `
var (
	_Num_indexes = [...]uint8{3, 6, 8, 10, 12}
	_Num_names   = "m_2m_1m0m1m2"
)

func (i Num) String() string {
	i -= -2
	if i < 0 || i >= Num(len(_Num_indexes)) {
		return fmt.Sprintf("Num(%d)", i+-2)
	}
	hi := _Num_indexes[i]
	lo := uint8(0)
	if i > 0 {
		lo = _Num_indexes[i-1]
	}
	return _Num_names[lo:hi]
}
`

// Unsigned integers spanning zero.
const uneg_in = `type UNum uint
const (
	m_2 UNum = ^UNum(0)-2
	m_1
	m0
	m1
	m2
)
`

const uneg_out = `
var (
	_UNum_indexes = [...]uint8{3}
	_UNum_names   = "m_2"
)

func (i UNum) String() string {
	i -= 18446744073709551613
	if i >= UNum(len(_UNum_indexes)) {
		return fmt.Sprintf("UNum(%d)", i+18446744073709551613)
	}
	hi := _UNum_indexes[i]
	lo := uint8(0)
	if i > 0 {
		lo = _UNum_indexes[i-1]
	}
	return _UNum_names[lo:hi]
}
`

// Enough gaps to trigger a map implementation of the method.
// Also includes a duplicate to test that it doesn't cause problems
const map_in = `type Prime int
const (
	p2 Prime = 2
	p3 Prime = 3
	p5 Prime = 5
	p7 Prime = 7
	p77 Prime = 7 // Duplicate; note that p77 doesn't appear below.
	p11 Prime = 11
	p13 Prime = 13
	p17 Prime = 17
	p19 Prime = 19
	p23 Prime = 23
	p29 Prime = 29
	p37 Prime = 31
	p41 Prime = 41
	p43 Prime = 43
)
`

const map_out = `
var _Prime_map = map[Prime]string{
	2:  "p2",
	3:  "p3",
	5:  "p5",
	7:  "p7",
	11: "p11",
	13: "p13",
	17: "p17",
	19: "p19",
	23: "p23",
	29: "p29",
	31: "p37",
	41: "p41",
	43: "p43",
}

func (i Prime) String() string {
	if str, ok := _Prime_map[i]; ok {
		return str
	}
	return fmt.Sprintf("Prime(%d)", i)
}
`

func TestGolden(t *testing.T) {
	for _, test := range golden {
		var g Generator
		input := "package test\n" + test.input
		file := test.name + ".go"
		g.parsePackage(".", []string{file}, input)
		// Extract the name and type of the constant from the first line.
		tokens := strings.SplitN(test.input, " ", 3)
		if len(tokens) != 3 {
			t.Fatalf("%s: need type declaration on first line", test.name)
		}
		g.generate(tokens[1])
		got := string(g.format())
		if got != test.output {
			t.Errorf("%s: got\n====\n%s====\nexpected\n====%s", test.name, got, test.output)
		}
	}
}
