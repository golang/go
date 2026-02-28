// run -gcflags=all=-d=checkptr

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"regexp"
	"unique"
)

var dataFileRegexp = regexp.MustCompile(`^data\.\d+\.bin$`)

func main() {
	_ = dataFileRegexp
	unique.Make("")
}
