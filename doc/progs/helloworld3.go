// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"file";
	"fmt";
)

func main() {
	hello := []byte{'h', 'e', 'l', 'l', 'o', ',', ' ', 'w', 'o', 'r', 'l', 'd', '\n'};
	file.Stdout.Write(hello);
	file, err := file.Open("/does/not/exist",  0,  0);
	if file == nil {
		fmt.Printf("can't open file; err=%s\n",  err.String());
		sys.Exit(1);
	}
}
