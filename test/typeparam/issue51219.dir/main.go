// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"a"
	"b"
	"fmt"
)

func main() {
	var x a.I[a.JsonRaw]
	var y b.InteractionRequest[a.JsonRaw]

	fmt.Printf("%v %v\n", x, y)
}
