// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import FD "fd"

func main() {
	hello := []byte{'h', 'e', 'l', 'l', 'o', ',', ' ', 'w', 'o', 'r', 'l', 'd', '\n'};
	FD.Stdout.Write(hello);
	fd,  errno := FD.Open("/does/not/exist",  0,  0);
	if fd == nil {
		print("can't open file; errno=",  errno,  "\n");
		sys.exit(1);
	}
}
