// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	nchar := 0;
	a := []int { '日', '本', '語', 0xFFFD };
	for _, char := range "日本語\xc0" {
		if nchar >= len(a) {
			println("BUG");
			break;
		}
		if char != a[nchar] {
			println("expected", a[nchar], "got", char);
			println("BUG");
			break;
		}
		nchar++;
	}
}
