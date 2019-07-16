// errorcheck

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that invalid imports are rejected by the compiler.
// Does not compile.

package main

// Correct import paths.
import _ "fmt"
import _ `time`
import _ "m\x61th"
import _ "go/parser"

// Correct import paths, but the packages don't exist.
// Don't test.
//import "a.b"
//import "greek/αβ"

// Import paths must be strings.
import 42    // ERROR "import path must be a string"
import 'a'   // ERROR "import path must be a string"
import 3.14  // ERROR "import path must be a string"
import 0.25i // ERROR "import path must be a string"
