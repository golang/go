// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"os";
	"regexp";
)

func main() {
	str := "a*b*c*";
	if sys.argc() > 1 {
		str = sys.argv(1);
	}
	re, err := regexp.Compile(str);
	if err != nil {
		print("error: ", err.String(), "\n");
		sys.exit(1);
	}
}
