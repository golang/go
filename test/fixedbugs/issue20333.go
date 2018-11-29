// compile

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 20333: early checkwidth of [...] arrays led to compilation errors.

package main

import "fmt"

func main() {
	fmt.Println(&[...]string{"abc", "def", "ghi"})
}
