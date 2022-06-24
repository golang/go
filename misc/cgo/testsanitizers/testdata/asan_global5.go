// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
)

type Any struct {
	s string
	b int64
}

var Sg = []interface{}{
	Any{"a", 10},
}

func main() {
	fmt.Println(Sg[0])
}
