// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

/*
const int sizeofLongDouble = sizeof(long double);
*/
import "C"

import "fmt"

func main() {
	fmt.Println(C.sizeofLongDouble)
}
