// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fd";
	"flag";
)

func cat(file *fd.FD) {
	const NBUF = 512;
	var buf [NBUF]byte;
	for {
		switch nr, er := file.Read(buf); true {
		case nr < 0:
			print("error reading from ", file.String(), ": ", er.String(), "\n");
			sys.Exit(1);
		case nr == 0:  // EOF
			return;
		case nr > 0:
			if nw, ew := fd.Stdout.Write(buf[0:nr]); nw != nr {
				print("error writing from ", file.String(), ": ", ew.String(), "\n");
			}
		}
	}
}

func main() {
	flag.Parse();   // Scans the arg list and sets up flags
	if flag.NArg() == 0 {
		cat(fd.Stdin);
	}
	for i := 0; i < flag.NArg(); i++ {
		file, err := fd.Open(flag.Arg(i), 0, 0);
		if file == nil {
			print("can't open ", flag.Arg(i), ": error ", err, "\n");
			sys.Exit(1);
		}
		cat(file);
		file.Close();
	}
}
