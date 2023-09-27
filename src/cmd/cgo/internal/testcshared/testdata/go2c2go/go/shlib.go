// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "C"

//export GoFunc
func GoFunc() int { return 1 }

func main() {}
