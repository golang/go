// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

import (
	"fmt"
)

//go:embed embedtext // want "must import \"embed\" when using go:embed directives"
var s string

// This is main function
func main() {
	fmt.Println(s)
}
