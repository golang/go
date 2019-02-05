// compile

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func ascii(r rune) rune {
	switch {
	case 97 <= r && r <= 122:
		return r - 32
	case 65 <= r && r <= 90:
		return r + 32
	default:
		return r
	}
}

func main() {
	nomeObjeto := "ABE1FK21"
	println(string(nomeObjeto[1:4]))
	println(ascii(rune(nomeObjeto[4])) >= 48 && ascii(rune(nomeObjeto[4])) <= 57)
	println(string(nomeObjeto[5]))
	println(string(nomeObjeto[6:10]))
}
