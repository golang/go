// errorcheckoutput

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test source files and strings containing NUL and invalid UTF-8.

package main

import (
	"fmt"
	"os"
)

func main() {
	var s = "\xc2\xff"
	var t = "\xd0\xfe"
	var u = "\xab\x00\xfc"

	if len(s) != 2 || s[0] != 0xc2 || s[1] != 0xff ||
		len(t) != 2 || t[0] != 0xd0 || t[1] != 0xfe ||
		len(u) != 3 || u[0] != 0xab || u[1] != 0x00 || u[2] != 0xfc {
		println("BUG: non-UTF-8 string mangled")
		os.Exit(2)
	}

	fmt.Print(`
package main

var x = "in string ` + "\x00" + `"	// ERROR "NUL"

var y = ` + "`in raw string \x00 foo`" + `  // ERROR "NUL"

// in comment ` + "\x00" + `  // ERROR "NUL"

/* in other comment ` + "\x00" + ` */ // ERROR "NUL"

/* in source code */ ` + "\x00" + `// ERROR "NUL" "illegal character"

var xx = "in string ` + "\xc2\xff" + `" // ERROR "UTF-8"

var yy = ` + "`in raw string \xff foo`" + `  // ERROR "UTF-8"

// in comment ` + "\xe2\x80\x01" + `  // ERROR "UTF-8"

/* in other comment ` + "\xe0\x00\x00" + ` */ // ERROR "UTF-8|NUL"

/* in variable name */
var z` + "\xc1\x81" + ` int // ERROR "UTF-8" "invalid identifier character"

/* in source code */ ` + "var \xc2A int" + `// ERROR "UTF-8" "invalid identifier character"

`)
}

