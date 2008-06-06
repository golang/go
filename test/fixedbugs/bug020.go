// $G $D/$F.go || echo BUG should compile

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
x.go :
main.go.c: In function ‘main_putint’:
main.go.c:41: error: syntax error before ‘)’ token
*/
