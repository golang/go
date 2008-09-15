// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	FD "fd";
	Flag "flag";
)

func cat(fd *FD.FD) {
	const NBUF = 512;
	var buf [NBUF]byte;
	for {
		switch nr, er := fd.Read(&buf); true {
		case nr < 0:
			print("error reading from ", fd.Name(), ": ", er, "\n");
			sys.exit(1);
		case nr == 0:  // EOF
			return;
		case nr > 0:
			if nw, ew := FD.Stdout.Write((&buf)[0:nr]); nw != nr {
				print("error writing from ", fd.Name(), ": ", ew, "\n");
			}
		}
	}
}

func main() {
	Flag.Parse();   // Scans the arg list and sets up flags
	if Flag.NArg() == 0 {
		cat(FD.Stdin);
	}
	for i := 0; i < Flag.NArg(); i++ {
		fd, err := FD.Open(Flag.Arg(i), 0, 0);
		if fd == nil {
			print("can't open ", Flag.Arg(i), ": error ", err, "\n");
			sys.exit(1);
		}
		cat(fd);
		fd.Close();
	}
}
