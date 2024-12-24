// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strings_test

import (
	"fmt"
	"strings"
	"unicode"
	"unsafe"
)

func ExampleClone() {
	s := "abc"
	clone := strings.Clone(s)
	fmt.Println(s == clone)
	fmt.Println(unsafe.StringData(s) == unsafe.StringData(clone))
	// Output:
	// true
	// false
}

func ExampleBuilder() {
	var b strings.Builder
	for i := 3; i >= 1; i-- {
		fmt.Fprintf(&b, "%d...", i)
	}
	b.WriteString("ignition")
	fmt.Println(b.String())

	// Output: 3...2...1...ignition
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
	fmt.Println(strings.ContainsAny("fail", "ui"))
	fmt.Println(strings.ContainsAny("ure", "ui"))
	fmt.Println(strings.ContainsAny("failure", "ui"))
	fmt.Println(strings.ContainsAny("foo", ""))
	fmt.Println(strings.ContainsAny("", ""))
	// Output:
	// false
	// true
	// true
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

func ExampleContainsFunc() {
	f := func(r rune) bool {
		return r == 'a' || r == 'e' || r == 'i' || r == 'o' || r == 'u'
	}
	fmt.Println(strings.ContainsFunc("hello", f))
	fmt.Println(strings.ContainsFunc("rhythms", f))
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

func ExampleCut() {
	show := func(s, sep string) {
		before, after, found := strings.Cut(s, sep)
		fmt.Printf("Cut(%q, %q) = %q, %q, %v\n", s, sep, before, after, found)
	}
	show("Gopher", "Go")
	show("Gopher", "ph")
	show("Gopher", "er")
	show("Gopher", "Badger")
	// Output:
	// Cut("Gopher", "Go") = "", "pher", true
	// Cut("Gopher", "ph") = "Go", "er", true
	// Cut("Gopher", "er") = "Goph", "", true
	// Cut("Gopher", "Badger") = "Gopher", "", false
}

func ExampleCutPrefix() {
	show := func(s, sep string) {
		after, found := strings.CutPrefix(s, sep)
		fmt.Printf("CutPrefix(%q, %q) = %q, %v\n", s, sep, after, found)
	}
	show("Gopher", "Go")
	show("Gopher", "ph")
	// Output:
	// CutPrefix("Gopher", "Go") = "pher", true
	// CutPrefix("Gopher", "ph") = "Gopher", false
}

func ExampleCutSuffix() {
	show := func(s, sep string) {
		before, found := strings.CutSuffix(s, sep)
		fmt.Printf("CutSuffix(%q, %q) = %q, %v\n", s, sep, before, found)
	}
	show("Gopher", "Go")
	show("Gopher", "er")
	// Output:
	// CutSuffix("Gopher", "Go") = "Gopher", false
	// CutSuffix("Gopher", "er") = "Goph", true
}

func ExampleEqualFold() {
	fmt.Println(strings.EqualFold("Go", "go"))
	fmt.Println(strings.EqualFold("AB", "ab")) // true because comparison uses simple case-folding
	fmt.Println(strings.EqualFold("ß", "ss"))  // false because comparison does not use full case-folding
	// Output:
	// true
	// true
	// false
}

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

func ExampleLastIndexByte() {
	fmt.Println(strings.LastIndexByte("Hello, world", 'l'))
	fmt.Println(strings.LastIndexByte("Hello, world", 'o'))
	fmt.Println(strings.LastIndexByte("Hello, world", 'x'))
	// Output:
	// 10
	// 8
	// -1
}

func ExampleLastIndexFunc() {
	fmt.Println(strings.LastIndexFunc("go 123", unicode.IsNumber))
	fmt.Println(strings.LastIndexFunc("123 go", unicode.IsNumber))
	fmt.Println(strings.LastIndexFunc("go", unicode.IsNumber))
	// Output:
	// 5
	// 2
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

func ExampleReplaceAll() {
	fmt.Println(strings.ReplaceAll("oink oink oink", "oink", "moo"))
	// Output:
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
	// Compare this example to the ToTitle example.
	fmt.Println(strings.Title("her royal highness"))
	fmt.Println(strings.Title("loud noises"))
	fmt.Println(strings.Title("брат"))
	// Output:
	// Her Royal Highness
	// Loud Noises
	// Брат
}

func ExampleToTitle() {
	// Compare this example to the Title example.
	fmt.Println(strings.ToTitle("her royal highness"))
	fmt.Println(strings.ToTitle("loud noises"))
	fmt.Println(strings.ToTitle("брат"))
	// Output:
	// HER ROYAL HIGHNESS
	// LOUD NOISES
	// БРАТ
}

func ExampleToTitleSpecial() {
	fmt.Println(strings.ToTitleSpecial(unicode.TurkishCase, "dünyanın ilk borsa yapısı Aizonai kabul edilir"))
	// Output:
	// DÜNYANIN İLK BORSA YAPISI AİZONAİ KABUL EDİLİR
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

func ExampleNewReplacer() {
	r := strings.NewReplacer("<", "&lt;", ">", "&gt;")
	fmt.Println(r.Replace("This is <b>HTML</b>!"))
	// Output: This is &lt;b&gt;HTML&lt;/b&gt;!
}

func ExampleToUpper() {
	fmt.Println(strings.ToUpper("Gopher"))
	// Output: GOPHER
}

func ExampleToUpperSpecial() {
	fmt.Println(strings.ToUpperSpecial(unicode.TurkishCase, "örnek iş"))
	// Output: ÖRNEK İŞ
}

func ExampleToLower() {
	fmt.Println(strings.ToLower("Gopher"))
	// Output: gopher
}

func ExampleToLowerSpecial() {
	fmt.Println(strings.ToLowerSpecial(unicode.TurkishCase, "Örnek İş"))
	// Output: örnek iş
}

func ExampleTrim() {
	fmt.Print(strings.Trim("¡¡¡Hello, Gophers!!!", "!¡"))
	// Output: Hello, Gophers
}

func ExampleTrimSpace() {
	fmt.Println(strings.TrimSpace(" \t\n Hello, Gophers \n\t\r\n"))
	// Output: Hello, Gophers
}

func ExampleTrimPrefix() {
	var s = "¡¡¡Hello, Gophers!!!"
	s = strings.TrimPrefix(s, "¡¡¡Hello, ")
	s = strings.TrimPrefix(s, "¡¡¡Howdy, ")
	fmt.Print(s)
	// Output: Gophers!!!
}

func ExampleTrimSuffix() {
	var s = "¡¡¡Hello, Gophers!!!"
	s = strings.TrimSuffix(s, ", Gophers!!!")
	s = strings.TrimSuffix(s, ", Marmots!!!")
	fmt.Print(s)
	// Output: ¡¡¡Hello
}

func ExampleTrimFunc() {
	fmt.Print(strings.TrimFunc("¡¡¡Hello, Gophers!!!", func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r)
	}))
	// Output: Hello, Gophers
}

func ExampleTrimLeft() {
	fmt.Print(strings.TrimLeft("¡¡¡Hello, Gophers!!!", "!¡"))
	// Output: Hello, Gophers!!!
}

func ExampleTrimLeftFunc() {
	fmt.Print(strings.TrimLeftFunc("¡¡¡Hello, Gophers!!!", func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r)
	}))
	// Output: Hello, Gophers!!!
}

func ExampleTrimRight() {
	fmt.Print(strings.TrimRight("¡¡¡Hello, Gophers!!!", "!¡"))
	// Output: ¡¡¡Hello, Gophers
}

func ExampleTrimRightFunc() {
	fmt.Print(strings.TrimRightFunc("¡¡¡Hello, Gophers!!!", func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r)
	}))
	// Output: ¡¡¡Hello, Gophers
}

func ExampleToValidUTF8() {
	fmt.Printf("%s\n", strings.ToValidUTF8("abc", "\uFFFD"))
	fmt.Printf("%s\n", strings.ToValidUTF8("a\xffb\xC0\xAFc\xff", ""))
	fmt.Printf("%s\n", strings.ToValidUTF8("\xed\xa0\x80", "abc"))
	// Output:
	// abc
	// abc
	// abc
}
