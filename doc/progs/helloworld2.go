// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import OS "os"    // this package contains features for basic I/O

func main() {
	OS.Stdout.WriteString("Hello, world; or Καλημέρα κόσμε; or こんにちは 世界\n");
}
