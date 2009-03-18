// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"file";
	"flag";
	"fmt";
	"os";
)

func cat(f *file.File) {
	const NBUF = 512;
	var buf [NBUF]byte;
	for {
		switch nr, er := f.Read(buf); true {
		case nr < 0:
			fmt.Fprintf(os.Stderr, "error reading from %s: %s\n", f.String(), er.String());
			sys.Exit(1);
		case nr == 0:  // EOF
			return;
		case nr > 0:
			if nw, ew := file.Stdout.Write(buf[0:nr]); nw != nr {
				fmt.Fprintf(os.Stderr, "error writing from %s: %s\n", f.String(), ew.String());
			}
		}
	}
}

func main() {
	flag.Parse();   // Scans the arg list and sets up flags
	if flag.NArg() == 0 {
		cat(file.Stdin);
	}
	for i := 0; i < flag.NArg(); i++ {
		f, err := file.Open(flag.Arg(i), 0, 0);
		if f == nil {
			fmt.Fprintf(os.Stderr, "can't open %s: error %s\n", flag.Arg(i), err);
			sys.Exit(1);
		}
		cat(f);
		f.Close();
	}
}
