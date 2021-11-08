// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package utf8_test

import (
	"fmt"
	"unicode/utf8"
)

func ExampleDecodeLastRune() {
	b := []byte("Hello, 世界")

	for len(b) > 0 {
		r, size := utf8.DecodeLastRune(b)
		fmt.Printf("%c %v\n", r, size)

		b = b[:len(b)-size]
	}
	// Output:
	// 界 3
	// 世 3
	//   1
	// , 1
	// o 1
	// l 1
	// l 1
	// e 1
	// H 1
}

func ExampleDecodeLastRuneInString() {
	str := "Hello, 世界"

	for len(str) > 0 {
		r, size := utf8.DecodeLastRuneInString(str)
		fmt.Printf("%c %v\n", r, size)

		str = str[:len(str)-size]
	}
	// Output:
	// 界 3
	// 世 3
	//   1
	// , 1
	// o 1
	// l 1
	// l 1
	// e 1
	// H 1

}

func ExampleDecodeRune() {
	b := []byte("Hello, 世界")

	for len(b) > 0 {
		r, size := utf8.DecodeRune(b)
		fmt.Printf("%c %v\n", r, size)

		b = b[size:]
	}
	// Output:
	// H 1
	// e 1
	// l 1
	// l 1
	// o 1
	// , 1
	//   1
	// 世 3
	// 界 3
}

func ExampleDecodeRuneInString() {
	str := "Hello, 世界"

	for len(str) > 0 {
		r, size := utf8.DecodeRuneInString(str)
		fmt.Printf("%c %v\n", r, size)

		str = str[size:]
	}
	// Output:
	// H 1
	// e 1
	// l 1
	// l 1
	// o 1
	// , 1
	//   1
	// 世 3
	// 界 3
}

func ExampleEncodeRune() {
	r := '世'
	buf := make([]byte, 3)

	n := utf8.EncodeRune(buf, r)

	fmt.Println(buf)
	fmt.Println(n)
	// Output:
	// [228 184 150]
	// 3
}

func ExampleEncodeRune_outOfRange() {
	runes := []rune{
		// Less than 0, out of range.
		-1,
		// Greater than 0x10FFFF, out of range.
		0x110000,
		// The Unicode replacement character.
		utf8.RuneError,
	}
	for i, c := range runes {
		buf := make([]byte, 3)
		size := utf8.EncodeRune(buf, c)
		fmt.Printf("%d: %d %[2]s %d\n", i, buf, size)
	}
	// Output:
	// 0: [239 191 189] � 3
	// 1: [239 191 189] � 3
	// 2: [239 191 189] � 3
}

func ExampleFullRune() {
	buf := []byte{228, 184, 150} // 世
	fmt.Println(utf8.FullRune(buf))
	fmt.Println(utf8.FullRune(buf[:2]))
	// Output:
	// true
	// false
}

func ExampleFullRuneInString() {
	str := "世"
	fmt.Println(utf8.FullRuneInString(str))
	fmt.Println(utf8.FullRuneInString(str[:2]))
	// Output:
	// true
	// false
}

func ExampleRuneCount() {
	buf := []byte("Hello, 世界")
	fmt.Println("bytes =", len(buf))
	fmt.Println("runes =", utf8.RuneCount(buf))
	// Output:
	// bytes = 13
	// runes = 9
}

func ExampleRuneCountInString() {
	str := "Hello, 世界"
	fmt.Println("bytes =", len(str))
	fmt.Println("runes =", utf8.RuneCountInString(str))
	// Output:
	// bytes = 13
	// runes = 9
}

func ExampleRuneLen() {
	fmt.Println(utf8.RuneLen('a'))
	fmt.Println(utf8.RuneLen('界'))
	// Output:
	// 1
	// 3
}

func ExampleRuneStart() {
	buf := []byte("a界")
	fmt.Println(utf8.RuneStart(buf[0]))
	fmt.Println(utf8.RuneStart(buf[1]))
	fmt.Println(utf8.RuneStart(buf[2]))
	// Output:
	// true
	// true
	// false
}

func ExampleValid() {
	valid := []byte("Hello, 世界")
	invalid := []byte{0xff, 0xfe, 0xfd}

	fmt.Println(utf8.Valid(valid))
	fmt.Println(utf8.Valid(invalid))
	// Output:
	// true
	// false
}

func ExampleValidRune() {
	valid := 'a'
	invalid := rune(0xfffffff)

	fmt.Println(utf8.ValidRune(valid))
	fmt.Println(utf8.ValidRune(invalid))
	// Output:
	// true
	// false
}

func ExampleValidString() {
	valid := "Hello, 世界"
	invalid := string([]byte{0xff, 0xfe, 0xfd})

	fmt.Println(utf8.ValidString(valid))
	fmt.Println(utf8.ValidString(invalid))
	// Output:
	// true
	// false
}

func ExampleAppendRune() {
	buf1 := utf8.AppendRune(nil, 0x10000)
	buf2 := utf8.AppendRune([]byte("init"), 0x10000)
	fmt.Println(string(buf1))
	fmt.Println(string(buf2))
	// Output:
	// 𐀀
	// init𐀀
}
