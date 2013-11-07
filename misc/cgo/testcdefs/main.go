// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "os"

func test() int32 // in main.c

func main() {
	os.Exit(int(test()))
}
