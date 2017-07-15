// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strings_test

import (
	"fmt"
	"strings"
	"unicode"
)

func ExampleFields() {
	fmt.Printf("Fields are: %q", strings.Fields("  foo bar  baz   "))
	// Output: Fields are: ["foo" "bar" "baz"]
}

func ExampleFieldsFunc() {
	f := func(c rune) bool {
		return !unicode.IsLetter(c) && !unicode.IsNumber(c)
	}
	fmt.Printf("Fields are: %q", strings.FieldsFunc("  foo1;bar2,baz3...", f))
	// Output: Fields are: ["foo1" "bar2" "baz3"]
}

func ExampleCompare() {
	fmt.Println(strings.Compare("a", "b"))
	fmt.Println(strings.Compare("a", "a"))
	fmt.Println(strings.Compare("b", "a"))
	// Output:
	// -1
	// 0
	// 1
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

func ExampleContainsRune() {
	// Finds whether a string contains a particular Unicode code point.
	// The code point for the lowercase letter "a", for example, is 97.
	fmt.Println(strings.ContainsRune("aardvark", 97))
	fmt.Println(strings.ContainsRune("timeout", 97))
	// Output:
	// true
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

func ExampleHasPrefix() {
	fmt.Println(strings.HasPrefix("Gopher", "Go"))
	fmt.Println(strings.HasPrefix("Gopher", "C"))
	fmt.Println(strings.HasPrefix("Gopher", ""))
	// Output:
	// true
	// false
	// true
}

func ExampleHasSuffix() {
	fmt.Println(strings.HasSuffix("Amigo", "go"))
	fmt.Println(strings.HasSuffix("Amigo", "O"))
	fmt.Println(strings.HasSuffix("Amigo", "Ami"))
	fmt.Println(strings.HasSuffix("Amigo", ""))
	// Output:
	// true
	// false
	// false
	// true
}

func ExampleIndex() {
	fmt.Println(strings.Index("chicken", "ken"))
	fmt.Println(strings.Index("chicken", "dmr"))
	// Output:
	// 4
	// -1
}

func ExampleIndexFunc() {
	f := func(c rune) bool {
		return unicode.Is(unicode.Han, c)
	}
	fmt.Println(strings.IndexFunc("Hello, 世界", f))
	fmt.Println(strings.IndexFunc("Hello, world", f))
	// Output:
	// 7
	// -1
}

func ExampleIndexAny() {
	fmt.Println(strings.IndexAny("chicken", "aeiouy"))
	fmt.Println(strings.IndexAny("crwth", "aeiouy"))
	// Output:
	// 2
	// -1
}

func ExampleIndexByte() {
	fmt.Println(strings.IndexByte("golang", 'g'))
	fmt.Println(strings.IndexByte("gophers", 'h'))
	fmt.Println(strings.IndexByte("golang", 'x'))
	// Output:
	// 0
	// 3
	// -1
}
func ExampleIndexRune() {
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

func ExampleLastIndexAny() {
	fmt.Println(strings.LastIndexAny("go gopher", "go"))
	fmt.Println(strings.LastIndexAny("go gopher", "rodent"))
	fmt.Println(strings.LastIndexAny("go gopher", "fail"))
	// Output:
	// 4
	// 8
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
	fmt.Printf("[%q]", strings.Trim(" !!! Achtung! Achtung! !!! ", "! "))
	// Output: ["Achtung! Achtung"]
}

func ExampleTrimFunc() {
	f := func(c rune) bool {
		return !unicode.IsLetter(c) && !unicode.IsNumber(c)
	}
	fmt.Printf("[%q]", strings.TrimFunc("  Achtung1! Achtung2,...", f))
	// Output: ["Achtung1! Achtung2"]
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

func ExampleTrimSuffix() {
	var s = "Hello, goodbye, etc!"
	s = strings.TrimSuffix(s, "goodbye, etc!")
	s = strings.TrimSuffix(s, "planet")
	fmt.Print(s, "world!")
	// Output: Hello, world!
}

func ExampleTrimPrefix() {
	var s = "Goodbye,, world!"
	s = strings.TrimPrefix(s, "Goodbye,")
	s = strings.TrimPrefix(s, "Howdy,")
	fmt.Print("Hello" + s)
	// Output: Hello, world!
}
