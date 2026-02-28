// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "C"

var sink []byte

//export GoFunction7
func GoFunction7() {
	sink = make([]byte, 4096)
}

func main() {
}
