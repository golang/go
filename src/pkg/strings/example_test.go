// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strings_test

import (
	"fmt"
	"strings"
)

// Fields are: ["foo" "bar" "baz"]
func ExampleFields() {
	fmt.Printf("Fields are: %q", strings.Fields("  foo bar  baz   "))
}

// true
// false
// true
// true
func ExampleContains() {
	fmt.Println(strings.Contains("seafood", "foo"))
	fmt.Println(strings.Contains("seafood", "bar"))
	fmt.Println(strings.Contains("seafood", ""))
	fmt.Println(strings.Contains("", ""))
}

// false
// true
// false
// false
func ExampleContainsAny() {
	fmt.Println(strings.ContainsAny("team", "i"))
	fmt.Println(strings.ContainsAny("failure", "u & i"))
	fmt.Println(strings.ContainsAny("foo", ""))
	fmt.Println(strings.ContainsAny("", ""))

}

// 3
// 5
func ExampleCount() {
	fmt.Println(strings.Count("cheese", "e"))
	fmt.Println(strings.Count("five", "")) // before & after each rune
}

// true
func ExampleEqualFold() {
	fmt.Println(strings.EqualFold("Go", "go"))
}

// 4
// -1
func ExampleIndex() {
	fmt.Println(strings.Index("chicken", "ken"))
	fmt.Println(strings.Index("chicken", "dmr"))
}

// 4
// -1
func ExampleRune() {
	fmt.Println(strings.IndexRune("chicken", 'k'))
	fmt.Println(strings.IndexRune("chicken", 'd'))
}

// 0
// 3
// -1
func ExampleLastIndex() {
	fmt.Println(strings.Index("go gopher", "go"))
	fmt.Println(strings.LastIndex("go gopher", "go"))
	fmt.Println(strings.LastIndex("go gopher", "rodent"))
}

// foo, bar, baz
func ExampleJoin() {
	s := []string{"foo", "bar", "baz"}
	fmt.Println(strings.Join(s, ", "))
}

// banana
func ExampleRepeat() {
	fmt.Println("ba" + strings.Repeat("na", 2))
}

// oinky oinky oink
// moo moo moo
func ExampleReplace() {
	fmt.Println(strings.Replace("oink oink oink", "k", "ky", 2))
	fmt.Println(strings.Replace("oink oink oink", "oink", "moo", -1))
}

// ["a" "b" "c"]
// ["" "man " "plan " "canal panama"]
// [" " "x" "y" "z" " "]
// [""]
func ExampleSplit() {
	fmt.Printf("%q\n", strings.Split("a,b,c", ","))
	fmt.Printf("%q\n", strings.Split("a man a plan a canal panama", "a "))
	fmt.Printf("%q\n", strings.Split(" xyz ", ""))
	fmt.Printf("%q\n", strings.Split("", "Bernardo O'Higgins"))
}

// ["a" "b,c"]
// [] (nil = true)
func ExampleSplitN() {
	fmt.Printf("%q\n", strings.SplitN("a,b,c", ",", 2))
	z := strings.SplitN("a,b,c", ",", 0)
	fmt.Printf("%q (nil = %v)\n", z, z == nil)
}

// ["a," "b," "c"]
func ExampleSplitAfter() {
	fmt.Printf("%q\n", strings.SplitAfter("a,b,c", ","))
}

// ["a," "b,c"]
func ExampleSplitAfterN() {
	fmt.Printf("%q\n", strings.SplitAfterN("a,b,c", ",", 2))
}

// Her Royal Highness
func ExampleTitle() {
	fmt.Println(strings.Title("her royal highness"))
}

// LOUD NOISES
// ХЛЕБ
func ExampleToTitle() {
	fmt.Println(strings.ToTitle("loud noises"))
	fmt.Println(strings.ToTitle("хлеб"))
}

// [Achtung]
func ExampleTrim() {
	fmt.Printf("[%s]", strings.Trim(" !!! Achtung !!! ", "! "))
}

// 'Gjnf oevyyvt naq gur fyvgul tbcure...
func ExampleMap() {
	rot13 := func(r rune) rune {
		switch {
		case r >= 'A' && r <= 'Z':
			return 'A' + (r-'A'+13)%26
		case r >= 'a' && r <= 'z':
			return 'a' + (r-'a'+13)%26
		}
		return r
	}
	fmt.Println(strings.Map(rot13, "'Twas brillig and the slithy gopher..."))
}

// a lone gopher
func ExampleTrimSpace() {
	fmt.Println(strings.TrimSpace(" \t\n a lone gopher \n\t\r\n"))
}

// This is &lt;b&gt;HTML&lt;/b&gt;!
func ExampleNewReplacer() {
	r := strings.NewReplacer("<", "&lt;", ">", "&gt;")
	fmt.Println(r.Replace("This is <b>HTML</b>!"))
}

// GOPHER
func ExampleToUpper() {
	fmt.Println(strings.ToUpper("Gopher"))
}

// gopher
func ExampleToLower() {
	fmt.Println(strings.ToLower("Gopher"))
}
