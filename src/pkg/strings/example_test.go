// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strings_test

import (
	"fmt"
	"strings"
)

func ExampleFields() {
	fmt.Printf("Fields are: %q", strings.Fields("  foo bar  baz   "))
	// Output: Fields are: ["foo" "bar" "baz"]
}

func ExampleContains() {
	fmt.Println(strings.Contains("seafood", "foo"))
	fmt.Println(strings.Contains("seafood", "bar"))
	fmt.Println(strings.Contains("seafood", ""))
	fmt.Println(strings.Contains("", ""))
	// Output:
	// true
	// false
	// true
	// true
}

func ExampleContainsAny() {
	fmt.Println(strings.ContainsAny("team", "i"))
	fmt.Println(strings.ContainsAny("failure", "u & i"))
	fmt.Println(strings.ContainsAny("foo", ""))
	fmt.Println(strings.ContainsAny("", ""))
	// Output:
	// false
	// true
	// false
	// false
}

func ExampleCount() {
	fmt.Println(strings.Count("cheese", "e"))
	fmt.Println(strings.Count("five", "")) // before & after each rune

	// Output:
	// 3
	// 5
}

func ExampleEqualFold() {
	fmt.Println(strings.EqualFold("Go", "go"))
	// Output: true
}

func ExampleIndex() {
	fmt.Println(strings.Index("chicken", "ken"))
	fmt.Println(strings.Index("chicken", "dmr"))
	// Output:
	// 4
	// -1
}

func ExampleRune() {
	fmt.Println(strings.IndexRune("chicken", 'k'))
	fmt.Println(strings.IndexRune("chicken", 'd'))
	// Output:
	// 4
	// -1
}

func ExampleLastIndex() {
	fmt.Println(strings.Index("go gopher", "go"))
	fmt.Println(strings.LastIndex("go gopher", "go"))
	fmt.Println(strings.LastIndex("go gopher", "rodent"))
	// Output:
	// 0
	// 3
	// -1
}

func ExampleJoin() {
	s := []string{"foo", "bar", "baz"}
	fmt.Println(strings.Join(s, ", "))
	// Output: foo, bar, baz
}

func ExampleRepeat() {
	fmt.Println("ba" + strings.Repeat("na", 2))
	// Output: banana
}

func ExampleReplace() {
	fmt.Println(strings.Replace("oink oink oink", "k", "ky", 2))
	fmt.Println(strings.Replace("oink oink oink", "oink", "moo", -1))
	// Output:
	// oinky oinky oink
	// moo moo moo
}

func ExampleSplit() {
	fmt.Printf("%q\n", strings.Split("a,b,c", ","))
	fmt.Printf("%q\n", strings.Split("a man a plan a canal panama", "a "))
	fmt.Printf("%q\n", strings.Split(" xyz ", ""))
	fmt.Printf("%q\n", strings.Split("", "Bernardo O'Higgins"))
	// Output:
	// ["a" "b" "c"]
	// ["" "man " "plan " "canal panama"]
	// [" " "x" "y" "z" " "]
	// [""]
}

func ExampleSplitN() {
	fmt.Printf("%q\n", strings.SplitN("a,b,c", ",", 2))
	z := strings.SplitN("a,b,c", ",", 0)
	fmt.Printf("%q (nil = %v)\n", z, z == nil)
	// Output:
	// ["a" "b,c"]
	// [] (nil = true)
}

func ExampleSplitAfter() {
	fmt.Printf("%q\n", strings.SplitAfter("a,b,c", ","))
	// Output: ["a," "b," "c"]
}

func ExampleSplitAfterN() {
	fmt.Printf("%q\n", strings.SplitAfterN("a,b,c", ",", 2))
	// Output: ["a," "b,c"]
}

func ExampleTitle() {
	fmt.Println(strings.Title("her royal highness"))
	// Output: Her Royal Highness
}

func ExampleToTitle() {
	fmt.Println(strings.ToTitle("loud noises"))
	fmt.Println(strings.ToTitle("хлеб"))
	// Output:
	// LOUD NOISES
	// ХЛЕБ
}

func ExampleTrim() {
	fmt.Printf("[%q]", strings.Trim(" !!! Achtung !!! ", "! "))
	// Output: ["Achtung"]
}

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
	// Output: 'Gjnf oevyyvt naq gur fyvgul tbcure...
}

func ExampleTrimSpace() {
	fmt.Println(strings.TrimSpace(" \t\n a lone gopher \n\t\r\n"))
	// Output: a lone gopher
}

func ExampleNewReplacer() {
	r := strings.NewReplacer("<", "&lt;", ">", "&gt;")
	fmt.Println(r.Replace("This is <b>HTML</b>!"))
	// Output: This is &lt;b&gt;HTML&lt;/b&gt;!
}

func ExampleToUpper() {
	fmt.Println(strings.ToUpper("Gopher"))
	// Output: GOPHER
}

func ExampleToLower() {
	fmt.Println(strings.ToLower("Gopher"))
	// Output: gopher
}
