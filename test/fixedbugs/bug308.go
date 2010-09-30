// $G $D/$F.go

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 1136

package main

import "fmt"

func log1(f string, argv ...interface{}) {
	fmt.Printf("log: %s\n", fmt.Sprintf(f, argv...))
}

func main() {
	log1("%d", 42)
}
