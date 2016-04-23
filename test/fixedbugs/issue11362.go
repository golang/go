// errorcheck

// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 11362: prints empty canonical import path

package main

import _ "unicode//utf8" // ERROR "non-canonical import path .unicode//utf8. \(should be .unicode/utf8.\)" "can't find import: .unicode//utf8."

func main() {
}

