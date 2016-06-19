// run

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "os"
import "./exp"

func main() {
	_ = exp.Exported(len(os.Args))
}
