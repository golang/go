// compile

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var digits string;

func putint(buf []byte, i, base, val int, digits string) {
		buf[i] = digits[val];
}

func main() {
}

/*
uetli:~/Source/go1/test gri$ 6g bugs/bug020.go
bugs/bug020.go:7: type of a structure field cannot be an open array
bugs/bug020.go:7: fatal error: width of a dynamic array
*/
