// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"

	"./Þfoo"
	Þblix "./Þfoo"
)

func main() {
	fmt.Printf("Þfoo.Þbar(33) returns %v\n", Þfoo.Þbar(33))
	fmt.Printf("Þblix.Þbar(33) returns %v\n", Þblix.Þbar(33))
}
