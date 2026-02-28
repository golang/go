// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Examples from the language spec section on string conversions.

package main

func main() {
	// 1
	_ = string('a')  // "a"
	_ = string(-1)   // "\ufffd" == "\xef\xbf\xbd"
	_ = string(0xf8) // "\u00f8" == "ø" == "\xc3\xb8"

	type myString string
	_ = myString(0x65e5) // "\u65e5" == "日" == "\xe6\x97\xa5"

	// 2
	_ = string([]byte{'h', 'e', 'l', 'l', '\xc3', '\xb8'}) // "hellø"
	_ = string([]byte{})                                   // ""
	_ = string([]byte(nil))                                // ""

	type bytes []byte
	_ = string(bytes{'h', 'e', 'l', 'l', '\xc3', '\xb8'}) // "hellø"

	type myByte byte
	_ = string([]myByte{'w', 'o', 'r', 'l', 'd', '!'})     // "world!"
	_ = myString([]myByte{'\xf0', '\x9f', '\x8c', '\x8d'}) // "🌍

	// 3
	_ = string([]rune{0x767d, 0x9d6c, 0x7fd4}) // "\u767d\u9d6c\u7fd4" == "白鵬翔"
	_ = string([]rune{})                       // ""
	_ = string([]rune(nil))                    // ""

	type runes []rune
	_ = string(runes{0x767d, 0x9d6c, 0x7fd4}) // "\u767d\u9d6c\u7fd4" == "白鵬翔"

	type myRune rune
	_ = string([]myRune{0x266b, 0x266c}) // "\u266b\u266c" == "♫♬"
	_ = myString([]myRune{0x1f30e})      // "\U0001f30e" == "🌎

	// 4
	_ = []byte("hellø") // []byte{'h', 'e', 'l', 'l', '\xc3', '\xb8'}
	_ = []byte("")      // []byte{}

	_ = bytes("hellø") // []byte{'h', 'e', 'l', 'l', '\xc3', '\xb8'}

	_ = []myByte("world!")      // []myByte{'w', 'o', 'r', 'l', 'd', '!'}
	_ = []myByte(myString("🌏")) // []myByte{'\xf0', '\x9f', '\x8c', '\x8f'}

	// 5
	_ = []rune(myString("白鵬翔")) // []rune{0x767d, 0x9d6c, 0x7fd4}
	_ = []rune("")              // []rune{}

	_ = runes("白鵬翔") // []rune{0x767d, 0x9d6c, 0x7fd4}

	_ = []myRune("♫♬")          // []myRune{0x266b, 0x266c}
	_ = []myRune(myString("🌐")) // []myRune{0x1f310}
}
