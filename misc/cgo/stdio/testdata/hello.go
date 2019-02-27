// run -tags=use_go_run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build test_run

package main

import "cgostdio/stdio"

func main() {
	stdio.Stdout.WriteString(stdio.Greeting + "\n")
}
