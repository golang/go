// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"
import "os"

func main() {
	s := "hello";
	if s[1] != 'e' {
		os.Exit(1);
	}
	s = "good bye";
	var p *string = &s;
	*p = "ciao";
}
