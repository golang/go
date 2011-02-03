// $G $D/$F.go && $L $F.$A && ./$A.out || echo BUG: bug321

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Troublesome floating point constants. Issue 1463.

package main

import "fmt"

func check(test string, got, want float64) bool {
	if got != want {
		fmt.Println(test, "got", got, "want", want)
		return false
	}
	return true
}

func main() {
	good := true
	// http://www.exploringbinary.com/java-hangs-when-converting-2-2250738585072012e-308/
	good = good && check("2.2250738585072012e-308", 2.2250738585072012e-308, 2.2250738585072014e-308)
	// http://www.exploringbinary.com/php-hangs-on-numeric-value-2-2250738585072011e-308/
	good = good && check("2.2250738585072011e-308", 2.2250738585072011e-308, 2.225073858507201e-308)
	if !good {
		panic("fail")
	}
}
