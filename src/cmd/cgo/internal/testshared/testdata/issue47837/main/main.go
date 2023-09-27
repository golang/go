// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"testshared/issue47837/a"
)

func main() {
	var vara a.ImplA
	a.TheFuncWithArgA(&vara)
}
