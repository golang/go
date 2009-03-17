// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import file "file"

func main() {
	hello := []byte{'h', 'e', 'l', 'l', 'o', ',', ' ', 'w', 'o', 'r', 'l', 'd', '\n'};
	file.Stdout.Write(hello);
	file, err := file.Open("/does/not/exist",  0,  0);
	if file == nil {
		print("can't open file; err=",  err.String(),  "\n");
		sys.Exit(1);
	}
}
