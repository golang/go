// run

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

var x = uint32(0x01020304)
var y = [...]uint32{1,2,3,4,5}

func main() {
	fmt.Sprint(y[byte(x)])
}
