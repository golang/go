// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unicode_test

import (
	"fmt"
	"unicode"
)

// Functions starting with "Is" can be used to inspect which table of range a
// rune belongs to. Note that runes may fit into more than one range.
func Example_is() {

	// constant with mixed type runes
	const mixed = "\b5Ὂg̀9! ℃ᾭG"
	for _, c := range mixed {
		fmt.Printf("For %q:\n", c)
		if unicode.IsControl(c) {
			fmt.Println("\tis control rune")
		}
		if unicode.IsDigit(c) {
			fmt.Println("\tis digit rune")
		}
		if unicode.IsGraphic(c) {
			fmt.Println("\tis graphic rune")
		}
		if unicode.IsLetter(c) {
			fmt.Println("\tis letter rune")
		}
		if unicode.IsLower(c) {
			fmt.Println("\tis lower case rune")
		}
		if unicode.IsMark(c) {
			fmt.Println("\tis mark rune")
		}
		if unicode.IsNumber(c) {
			fmt.Println("\tis number rune")
		}
		if unicode.IsPrint(c) {
			fmt.Println("\tis printable rune")
		}
		if !unicode.IsPrint(c) {
			fmt.Println("\tis not printable rune")
		}
		if unicode.IsPunct(c) {
			fmt.Println("\tis punct rune")
		}
		if unicode.IsSpace(c) {
			fmt.Println("\tis space rune")
		}
		if unicode.IsSymbol(c) {
			fmt.Println("\tis symbol rune")
		}
		if unicode.IsTitle(c) {
			fmt.Println("\tis title case rune")
		}
		if unicode.IsUpper(c) {
			fmt.Println("\tis upper case rune")
		}
	}

	// Output:
	// For '\b':
	// 	is control rune
	// 	is not printable rune
	// For '5':
	// 	is digit rune
	// 	is graphic rune
	// 	is number rune
	// 	is printable rune
	// For 'Ὂ':
	// 	is graphic rune
	// 	is letter rune
	// 	is printable rune
	// 	is upper case rune
	// For 'g':
	// 	is graphic rune
	// 	is letter rune
	// 	is lower case rune
	// 	is printable rune
	// For '̀':
	// 	is graphic rune
	// 	is mark rune
	// 	is printable rune
	// For '9':
	// 	is digit rune
	// 	is graphic rune
	// 	is number rune
	// 	is printable rune
	// For '!':
	// 	is graphic rune
	// 	is printable rune
	// 	is punct rune
	// For ' ':
	// 	is graphic rune
	// 	is printable rune
	// 	is space rune
	// For '℃':
	// 	is graphic rune
	// 	is printable rune
	// 	is symbol rune
	// For 'ᾭ':
	// 	is graphic rune
	// 	is letter rune
	// 	is printable rune
	// 	is title case rune
	// For 'G':
	// 	is graphic rune
	// 	is letter rune
	// 	is printable rune
	// 	is upper case rune
}

func ExampleSimpleFold() {
	fmt.Printf("%#U\n", unicode.SimpleFold('A'))      // 'a'
	fmt.Printf("%#U\n", unicode.SimpleFold('a'))      // 'A'
	fmt.Printf("%#U\n", unicode.SimpleFold('K'))      // 'k'
	fmt.Printf("%#U\n", unicode.SimpleFold('k'))      // '\u212A' (Kelvin symbol, K)
	fmt.Printf("%#U\n", unicode.SimpleFold('\u212A')) // 'K'
	fmt.Printf("%#U\n", unicode.SimpleFold('1'))      // '1'

	// Output:
	// U+0061 'a'
	// U+0041 'A'
	// U+006B 'k'
	// U+212A 'K'
	// U+004B 'K'
	// U+0031 '1'
}

func ExampleTo() {
	const lcG = 'g'
	fmt.Printf("%#U\n", unicode.To(unicode.UpperCase, lcG))
	fmt.Printf("%#U\n", unicode.To(unicode.LowerCase, lcG))
	fmt.Printf("%#U\n", unicode.To(unicode.TitleCase, lcG))

	const ucG = 'G'
	fmt.Printf("%#U\n", unicode.To(unicode.UpperCase, ucG))
	fmt.Printf("%#U\n", unicode.To(unicode.LowerCase, ucG))
	fmt.Printf("%#U\n", unicode.To(unicode.TitleCase, ucG))

	// Output:
	// U+0047 'G'
	// U+0067 'g'
	// U+0047 'G'
	// U+0047 'G'
	// U+0067 'g'
	// U+0047 'G'
}

func ExampleToLower() {
	const ucG = 'G'
	fmt.Printf("%#U\n", unicode.ToLower(ucG))

	// Output:
	// U+0067 'g'
}
func ExampleToTitle() {
	const ucG = 'g'
	fmt.Printf("%#U\n", unicode.ToTitle(ucG))

	// Output:
	// U+0047 'G'
}

func ExampleToUpper() {
	const ucG = 'g'
	fmt.Printf("%#U\n", unicode.ToUpper(ucG))

	// Output:
	// U+0047 'G'
}

func ExampleSpecialCase() {
	t := unicode.TurkishCase

	const lci = 'i'
	fmt.Printf("%#U\n", t.ToLower(lci))
	fmt.Printf("%#U\n", t.ToTitle(lci))
	fmt.Printf("%#U\n", t.ToUpper(lci))

	const uci = 'İ'
	fmt.Printf("%#U\n", t.ToLower(uci))
	fmt.Printf("%#U\n", t.ToTitle(uci))
	fmt.Printf("%#U\n", t.ToUpper(uci))

	// Output:
	// U+0069 'i'
	// U+0130 'İ'
	// U+0130 'İ'
	// U+0069 'i'
	// U+0130 'İ'
	// U+0130 'İ'
}
