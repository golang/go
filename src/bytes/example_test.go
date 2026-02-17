// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bytes_test

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"io"
	"os"
	"slices"
	"strconv"
	"unicode"
)

func ExampleBuffer() {
	var b bytes.Buffer // A Buffer needs no initialization.
	b.Write([]byte("Hello "))
	fmt.Fprintf(&b, "world!")
	b.WriteTo(os.Stdout)
	// Output: Hello world!
}

func ExampleBuffer_reader() {
	// A Buffer can turn a string or a []byte into an io.Reader.
	buf := bytes.NewBufferString("R29waGVycyBydWxlIQ==")
	dec := base64.NewDecoder(base64.StdEncoding, buf)
	io.Copy(os.Stdout, dec)
	// Output: Gophers rule!
}

func ExampleBuffer_Bytes() {
	buf := bytes.Buffer{}
	buf.Write([]byte{'h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd'})
	os.Stdout.Write(buf.Bytes())
	// Output: hello world
}

func ExampleBuffer_AvailableBuffer() {
	var buf bytes.Buffer
	for i := 0; i < 4; i++ {
		b := buf.AvailableBuffer()
		b = strconv.AppendInt(b, int64(i), 10)
		b = append(b, ' ')
		buf.Write(b)
	}
	os.Stdout.Write(buf.Bytes())
	// Output: 0 1 2 3
}

func ExampleBuffer_Cap() {
	buf1 := bytes.NewBuffer(make([]byte, 10))
	buf2 := bytes.NewBuffer(make([]byte, 0, 10))
	fmt.Println(buf1.Cap())
	fmt.Println(buf2.Cap())
	// Output:
	// 10
	// 10
}

func ExampleBuffer_Grow() {
	var b bytes.Buffer
	b.Grow(64)
	bb := b.Bytes()
	b.Write([]byte("64 bytes or fewer"))
	fmt.Printf("%q", bb[:b.Len()])
	// Output: "64 bytes or fewer"
}

func ExampleBuffer_Len() {
	var b bytes.Buffer
	b.Grow(64)
	b.Write([]byte("abcde"))
	fmt.Printf("%d", b.Len())
	// Output: 5
}

func ExampleBuffer_Next() {
	var b bytes.Buffer
	b.Grow(64)
	b.Write([]byte("abcde"))
	fmt.Printf("%s\n", b.Next(2))
	fmt.Printf("%s\n", b.Next(2))
	fmt.Printf("%s", b.Next(2))
	// Output:
	// ab
	// cd
	// e
}

func ExampleBuffer_Read() {
	var b bytes.Buffer
	b.Grow(64)
	b.Write([]byte("abcde"))
	rdbuf := make([]byte, 1)
	n, err := b.Read(rdbuf)
	if err != nil {
		panic(err)
	}
	fmt.Println(n)
	fmt.Println(b.String())
	fmt.Println(string(rdbuf))
	// Output:
	// 1
	// bcde
	// a
}

func ExampleBuffer_ReadByte() {
	var b bytes.Buffer
	b.Grow(64)
	b.Write([]byte("abcde"))
	c, err := b.ReadByte()
	if err != nil {
		panic(err)
	}
	fmt.Println(c)
	fmt.Println(b.String())
	// Output:
	// 97
	// bcde
}

func ExampleClone() {
	b := []byte("abc")
	clone := bytes.Clone(b)
	fmt.Printf("%s\n", clone)
	clone[0] = 'd'
	fmt.Printf("%s\n", b)
	fmt.Printf("%s\n", clone)
	// Output:
	// abc
	// abc
	// dbc
}

func ExampleCompare() {
	// Interpret Compare's result by comparing it to zero.
	var a, b []byte
	if bytes.Compare(a, b) < 0 {
		// a less b
	}
	if bytes.Compare(a, b) <= 0 {
		// a less or equal b
	}
	if bytes.Compare(a, b) > 0 {
		// a greater b
	}
	if bytes.Compare(a, b) >= 0 {
		// a greater or equal b
	}

	// Prefer Equal to Compare for equality comparisons.
	if bytes.Equal(a, b) {
		// a equal b
	}
	if !bytes.Equal(a, b) {
		// a not equal b
	}
}

func ExampleCompare_search() {
	// Binary search to find a matching byte slice.
	var needle []byte
	var haystack [][]byte // Assume sorted
	_, found := slices.BinarySearchFunc(haystack, needle, bytes.Compare)
	if found {
		// Found it!
	}
}

func ExampleContains() {
	fmt.Println(bytes.Contains([]byte("seafood"), []byte("foo")))
	fmt.Println(bytes.Contains([]byte("seafood"), []byte("bar")))
	fmt.Println(bytes.Contains([]byte("seafood"), []byte("")))
	fmt.Println(bytes.Contains([]byte(""), []byte("")))
	// Output:
	// true
	// false
	// true
	// true
}

func ExampleContainsAny() {
	fmt.Println(bytes.ContainsAny([]byte("I like seafood."), "fÄo!"))
	fmt.Println(bytes.ContainsAny([]byte("I like seafood."), "去是伟大的."))
	fmt.Println(bytes.ContainsAny([]byte("I like seafood."), ""))
	fmt.Println(bytes.ContainsAny([]byte(""), ""))
	// Output:
	// true
	// true
	// false
	// false
}

func ExampleContainsRune() {
	fmt.Println(bytes.ContainsRune([]byte("I like seafood."), 'f'))
	fmt.Println(bytes.ContainsRune([]byte("I like seafood."), 'ö'))
	fmt.Println(bytes.ContainsRune([]byte("去是伟大的!"), '大'))
	fmt.Println(bytes.ContainsRune([]byte("去是伟大的!"), '!'))
	fmt.Println(bytes.ContainsRune([]byte(""), '@'))
	// Output:
	// true
	// false
	// true
	// true
	// false
}

func ExampleContainsFunc() {
	f := func(r rune) bool {
		return r >= 'a' && r <= 'z'
	}
	fmt.Println(bytes.ContainsFunc([]byte("HELLO"), f))
	fmt.Println(bytes.ContainsFunc([]byte("World"), f))
	// Output:
	// false
	// true
}

func ExampleCount() {
	fmt.Println(bytes.Count([]byte("cheese"), []byte("e")))
	fmt.Println(bytes.Count([]byte("five"), []byte(""))) // before & after each rune
	// Output:
	// 3
	// 5
}

func ExampleCut() {
	show := func(s, sep string) {
		before, after, found := bytes.Cut([]byte(s), []byte(sep))
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
	show := func(s, prefix string) {
		after, found := bytes.CutPrefix([]byte(s), []byte(prefix))
		fmt.Printf("CutPrefix(%q, %q) = %q, %v\n", s, prefix, after, found)
	}
	show("Gopher", "Go")
	show("Gopher", "ph")
	// Output:
	// CutPrefix("Gopher", "Go") = "pher", true
	// CutPrefix("Gopher", "ph") = "Gopher", false
}

func ExampleCutSuffix() {
	show := func(s, suffix string) {
		before, found := bytes.CutSuffix([]byte(s), []byte(suffix))
		fmt.Printf("CutSuffix(%q, %q) = %q, %v\n", s, suffix, before, found)
	}
	show("Gopher", "Go")
	show("Gopher", "er")
	// Output:
	// CutSuffix("Gopher", "Go") = "Gopher", false
	// CutSuffix("Gopher", "er") = "Goph", true
}

func ExampleEqual() {
	fmt.Println(bytes.Equal([]byte("Go"), []byte("Go")))
	fmt.Println(bytes.Equal([]byte("Go"), []byte("C++")))
	// Output:
	// true
	// false
}

func ExampleEqualFold() {
	fmt.Println(bytes.EqualFold([]byte("Go"), []byte("go")))
	// Output: true
}

func ExampleFields() {
	fmt.Printf("Fields are: %q", bytes.Fields([]byte("  foo bar  baz   ")))
	// Output: Fields are: ["foo" "bar" "baz"]
}

func ExampleFieldsFunc() {
	f := func(c rune) bool {
		return !unicode.IsLetter(c) && !unicode.IsNumber(c)
	}
	fmt.Printf("Fields are: %q", bytes.FieldsFunc([]byte("  foo1;bar2,baz3..."), f))
	// Output: Fields are: ["foo1" "bar2" "baz3"]
}

func ExampleHasPrefix() {
	fmt.Println(bytes.HasPrefix([]byte("Gopher"), []byte("Go")))
	fmt.Println(bytes.HasPrefix([]byte("Gopher"), []byte("C")))
	fmt.Println(bytes.HasPrefix([]byte("Gopher"), []byte("")))
	// Output:
	// true
	// false
	// true
}

func ExampleHasSuffix() {
	fmt.Println(bytes.HasSuffix([]byte("Amigo"), []byte("go")))
	fmt.Println(bytes.HasSuffix([]byte("Amigo"), []byte("O")))
	fmt.Println(bytes.HasSuffix([]byte("Amigo"), []byte("Ami")))
	fmt.Println(bytes.HasSuffix([]byte("Amigo"), []byte("")))
	// Output:
	// true
	// false
	// false
	// true
}

func ExampleIndex() {
	fmt.Println(bytes.Index([]byte("chicken"), []byte("ken")))
	fmt.Println(bytes.Index([]byte("chicken"), []byte("dmr")))
	// Output:
	// 4
	// -1
}

func ExampleIndexByte() {
	fmt.Println(bytes.IndexByte([]byte("chicken"), byte('k')))
	fmt.Println(bytes.IndexByte([]byte("chicken"), byte('g')))
	// Output:
	// 4
	// -1
}

func ExampleIndexFunc() {
	f := func(c rune) bool {
		return unicode.Is(unicode.Han, c)
	}
	fmt.Println(bytes.IndexFunc([]byte("Hello, 世界"), f))
	fmt.Println(bytes.IndexFunc([]byte("Hello, world"), f))
	// Output:
	// 7
	// -1
}

func ExampleIndexAny() {
	fmt.Println(bytes.IndexAny([]byte("chicken"), "aeiouy"))
	fmt.Println(bytes.IndexAny([]byte("crwth"), "aeiouy"))
	// Output:
	// 2
	// -1
}

func ExampleIndexRune() {
	fmt.Println(bytes.IndexRune([]byte("chicken"), 'k'))
	fmt.Println(bytes.IndexRune([]byte("chicken"), 'd'))
	// Output:
	// 4
	// -1
}

func ExampleJoin() {
	s := [][]byte{[]byte("foo"), []byte("bar"), []byte("baz")}
	fmt.Printf("%s", bytes.Join(s, []byte(", ")))
	// Output: foo, bar, baz
}

func ExampleLastIndex() {
	fmt.Println(bytes.Index([]byte("go gopher"), []byte("go")))
	fmt.Println(bytes.LastIndex([]byte("go gopher"), []byte("go")))
	fmt.Println(bytes.LastIndex([]byte("go gopher"), []byte("rodent")))
	// Output:
	// 0
	// 3
	// -1
}

func ExampleLastIndexAny() {
	fmt.Println(bytes.LastIndexAny([]byte("go gopher"), "MüQp"))
	fmt.Println(bytes.LastIndexAny([]byte("go 地鼠"), "地大"))
	fmt.Println(bytes.LastIndexAny([]byte("go gopher"), "z,!."))
	// Output:
	// 5
	// 3
	// -1
}

func ExampleLastIndexByte() {
	fmt.Println(bytes.LastIndexByte([]byte("go gopher"), byte('g')))
	fmt.Println(bytes.LastIndexByte([]byte("go gopher"), byte('r')))
	fmt.Println(bytes.LastIndexByte([]byte("go gopher"), byte('z')))
	// Output:
	// 3
	// 8
	// -1
}

func ExampleLastIndexFunc() {
	fmt.Println(bytes.LastIndexFunc([]byte("go gopher!"), unicode.IsLetter))
	fmt.Println(bytes.LastIndexFunc([]byte("go gopher!"), unicode.IsPunct))
	fmt.Println(bytes.LastIndexFunc([]byte("go gopher!"), unicode.IsNumber))
	// Output:
	// 8
	// 9
	// -1
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
	fmt.Printf("%s\n", bytes.Map(rot13, []byte("'Twas brillig and the slithy gopher...")))
	// Output:
	// 'Gjnf oevyyvt naq gur fyvgul tbcure...
}

func ExampleReader_Len() {
	fmt.Println(bytes.NewReader([]byte("Hi!")).Len())
	fmt.Println(bytes.NewReader([]byte("こんにちは!")).Len())
	// Output:
	// 3
	// 16
}

func ExampleRepeat() {
	fmt.Printf("ba%s", bytes.Repeat([]byte("na"), 2))
	// Output: banana
}

func ExampleReplace() {
	fmt.Printf("%s\n", bytes.Replace([]byte("oink oink oink"), []byte("k"), []byte("ky"), 2))
	fmt.Printf("%s\n", bytes.Replace([]byte("oink oink oink"), []byte("oink"), []byte("moo"), -1))
	// Output:
	// oinky oinky oink
	// moo moo moo
}

func ExampleReplaceAll() {
	fmt.Printf("%s\n", bytes.ReplaceAll([]byte("oink oink oink"), []byte("oink"), []byte("moo")))
	// Output:
	// moo moo moo
}

func ExampleRunes() {
	rs := bytes.Runes([]byte("go gopher"))
	for _, r := range rs {
		fmt.Printf("%#U\n", r)
	}
	// Output:
	// U+0067 'g'
	// U+006F 'o'
	// U+0020 ' '
	// U+0067 'g'
	// U+006F 'o'
	// U+0070 'p'
	// U+0068 'h'
	// U+0065 'e'
	// U+0072 'r'
}

func ExampleSplit() {
	fmt.Printf("%q\n", bytes.Split([]byte("a,b,c"), []byte(",")))
	fmt.Printf("%q\n", bytes.Split([]byte("a man a plan a canal panama"), []byte("a ")))
	fmt.Printf("%q\n", bytes.Split([]byte(" xyz "), []byte("")))
	fmt.Printf("%q\n", bytes.Split([]byte(""), []byte("Bernardo O'Higgins")))
	// Output:
	// ["a" "b" "c"]
	// ["" "man " "plan " "canal panama"]
	// [" " "x" "y" "z" " "]
	// [""]
}

func ExampleSplitN() {
	fmt.Printf("%q\n", bytes.SplitN([]byte("a,b,c"), []byte(","), 2))
	z := bytes.SplitN([]byte("a,b,c"), []byte(","), 0)
	fmt.Printf("%q (nil = %v)\n", z, z == nil)
	// Output:
	// ["a" "b,c"]
	// [] (nil = true)
}

func ExampleSplitAfter() {
	fmt.Printf("%q\n", bytes.SplitAfter([]byte("a,b,c"), []byte(",")))
	// Output: ["a," "b," "c"]
}

func ExampleSplitAfterN() {
	fmt.Printf("%q\n", bytes.SplitAfterN([]byte("a,b,c"), []byte(","), 2))
	// Output: ["a," "b,c"]
}

func ExampleTitle() {
	fmt.Printf("%s", bytes.Title([]byte("her royal highness")))
	// Output: Her Royal Highness
}

func ExampleToTitle() {
	fmt.Printf("%s\n", bytes.ToTitle([]byte("loud noises")))
	fmt.Printf("%s\n", bytes.ToTitle([]byte("брат")))
	// Output:
	// LOUD NOISES
	// БРАТ
}

func ExampleToTitleSpecial() {
	str := []byte("ahoj vývojári golang")
	totitle := bytes.ToTitleSpecial(unicode.AzeriCase, str)
	fmt.Println("Original : " + string(str))
	fmt.Println("ToTitle : " + string(totitle))
	// Output:
	// Original : ahoj vývojári golang
	// ToTitle : AHOJ VÝVOJÁRİ GOLANG
}

func ExampleToValidUTF8() {
	fmt.Printf("%s\n", bytes.ToValidUTF8([]byte("abc"), []byte("\uFFFD")))
	fmt.Printf("%s\n", bytes.ToValidUTF8([]byte("a\xffb\xC0\xAFc\xff"), []byte("")))
	fmt.Printf("%s\n", bytes.ToValidUTF8([]byte("\xed\xa0\x80"), []byte("abc")))
	// Output:
	// abc
	// abc
	// abc
}

func ExampleTrim() {
	fmt.Printf("[%q]", bytes.Trim([]byte(" !!! Achtung! Achtung! !!! "), "! "))
	// Output: ["Achtung! Achtung"]
}

func ExampleTrimFunc() {
	fmt.Println(string(bytes.TrimFunc([]byte("go-gopher!"), unicode.IsLetter)))
	fmt.Println(string(bytes.TrimFunc([]byte("\"go-gopher!\""), unicode.IsLetter)))
	fmt.Println(string(bytes.TrimFunc([]byte("go-gopher!"), unicode.IsPunct)))
	fmt.Println(string(bytes.TrimFunc([]byte("1234go-gopher!567"), unicode.IsNumber)))
	// Output:
	// -gopher!
	// "go-gopher!"
	// go-gopher
	// go-gopher!
}

func ExampleTrimLeft() {
	fmt.Print(string(bytes.TrimLeft([]byte("453gopher8257"), "0123456789")))
	// Output:
	// gopher8257
}

func ExampleTrimLeftFunc() {
	fmt.Println(string(bytes.TrimLeftFunc([]byte("go-gopher"), unicode.IsLetter)))
	fmt.Println(string(bytes.TrimLeftFunc([]byte("go-gopher!"), unicode.IsPunct)))
	fmt.Println(string(bytes.TrimLeftFunc([]byte("1234go-gopher!567"), unicode.IsNumber)))
	// Output:
	// -gopher
	// go-gopher!
	// go-gopher!567
}

func ExampleTrimPrefix() {
	var b = []byte("Goodbye,, world!")
	b = bytes.TrimPrefix(b, []byte("Goodbye,"))
	b = bytes.TrimPrefix(b, []byte("See ya,"))
	fmt.Printf("Hello%s", b)
	// Output: Hello, world!
}

func ExampleTrimSpace() {
	fmt.Printf("%s", bytes.TrimSpace([]byte(" \t\n a lone gopher \n\t\r\n")))
	// Output: a lone gopher
}

func ExampleTrimSuffix() {
	var b = []byte("Hello, goodbye, etc!")
	b = bytes.TrimSuffix(b, []byte("goodbye, etc!"))
	b = bytes.TrimSuffix(b, []byte("gopher"))
	b = append(b, bytes.TrimSuffix([]byte("world!"), []byte("x!"))...)
	os.Stdout.Write(b)
	// Output: Hello, world!
}

func ExampleTrimRight() {
	fmt.Print(string(bytes.TrimRight([]byte("453gopher8257"), "0123456789")))
	// Output:
	// 453gopher
}

func ExampleTrimRightFunc() {
	fmt.Println(string(bytes.TrimRightFunc([]byte("go-gopher"), unicode.IsLetter)))
	fmt.Println(string(bytes.TrimRightFunc([]byte("go-gopher!"), unicode.IsPunct)))
	fmt.Println(string(bytes.TrimRightFunc([]byte("1234go-gopher!567"), unicode.IsNumber)))
	// Output:
	// go-
	// go-gopher
	// 1234go-gopher!
}

func ExampleToLower() {
	fmt.Printf("%s", bytes.ToLower([]byte("Gopher")))
	// Output: gopher
}

func ExampleToLowerSpecial() {
	str := []byte("AHOJ VÝVOJÁRİ GOLANG")
	totitle := bytes.ToLowerSpecial(unicode.AzeriCase, str)
	fmt.Println("Original : " + string(str))
	fmt.Println("ToLower : " + string(totitle))
	// Output:
	// Original : AHOJ VÝVOJÁRİ GOLANG
	// ToLower : ahoj vývojári golang
}

func ExampleToUpper() {
	fmt.Printf("%s", bytes.ToUpper([]byte("Gopher")))
	// Output: GOPHER
}

func ExampleToUpperSpecial() {
	str := []byte("ahoj vývojári golang")
	totitle := bytes.ToUpperSpecial(unicode.AzeriCase, str)
	fmt.Println("Original : " + string(str))
	fmt.Println("ToUpper : " + string(totitle))
	// Output:
	// Original : ahoj vývojári golang
	// ToUpper : AHOJ VÝVOJÁRİ GOLANG
}

func ExampleLines() {
	text := []byte("Hello\nWorld\nGo Programming\n")
	for line := range bytes.Lines(text) {
		fmt.Printf("%q\n", line)
	}

	// Output:
	// "Hello\n"
	// "World\n"
	// "Go Programming\n"
}

func ExampleSplitSeq() {
	s := []byte("a,b,c,d")
	for part := range bytes.SplitSeq(s, []byte(",")) {
		fmt.Printf("%q\n", part)
	}

	// Output:
	// "a"
	// "b"
	// "c"
	// "d"
}

func ExampleSplitAfterSeq() {
	s := []byte("a,b,c,d")
	for part := range bytes.SplitAfterSeq(s, []byte(",")) {
		fmt.Printf("%q\n", part)
	}

	// Output:
	// "a,"
	// "b,"
	// "c,"
	// "d"
}

func ExampleFieldsSeq() {
	text := []byte("The quick brown fox")
	fmt.Println("Split byte slice into fields:")
	for word := range bytes.FieldsSeq(text) {
		fmt.Printf("%q\n", word)
	}

	textWithSpaces := []byte("  lots   of   spaces  ")
	fmt.Println("\nSplit byte slice with multiple spaces:")
	for word := range bytes.FieldsSeq(textWithSpaces) {
		fmt.Printf("%q\n", word)
	}

	// Output:
	// Split byte slice into fields:
	// "The"
	// "quick"
	// "brown"
	// "fox"
	//
	// Split byte slice with multiple spaces:
	// "lots"
	// "of"
	// "spaces"
}

func ExampleFieldsFuncSeq() {
	text := []byte("The quick brown fox")
	fmt.Println("Split on whitespace(similar to FieldsSeq):")
	for word := range bytes.FieldsFuncSeq(text, unicode.IsSpace) {
		fmt.Printf("%q\n", word)
	}

	mixedText := []byte("abc123def456ghi")
	fmt.Println("\nSplit on digits:")
	for word := range bytes.FieldsFuncSeq(mixedText, unicode.IsDigit) {
		fmt.Printf("%q\n", word)
	}

	// Output:
	// Split on whitespace(similar to FieldsSeq):
	// "The"
	// "quick"
	// "brown"
	// "fox"
	//
	// Split on digits:
	// "abc"
	// "def"
	// "ghi"
}
