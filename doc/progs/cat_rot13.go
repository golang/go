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

var rot13_flag = flag.Bool("rot13", false, "rot13 the input")

func rot13(b byte) byte {
	if 'a' <= b && b <= 'z' {
	   b = 'a' + ((b - 'a') + 13) % 26;
	}
	if 'A' <= b && b <= 'Z' {
	   b = 'A' + ((b - 'A') + 13) % 26
	}
	return b
}

type reader interface {
	Read(b []byte) (ret int, err os.Error);
	String() string;
}

type rotate13 struct {
	source	reader;
}

func newRotate13(source reader) *rotate13 {
	return &rotate13{source}
}

func (r13 *rotate13) Read(b []byte) (ret int, err os.Error) {
	r, e := r13.source.Read(b);
	for i := 0; i < r; i++ {
		b[i] = rot13(b[i])
	}
	return r, e
}

func (r13 *rotate13) String() string {
	return r13.source.String()
}
// end of rotate13 implementation

func cat(r reader) {
	const NBUF = 512;
	var buf [NBUF]byte;

	if *rot13_flag {
		r = newRotate13(r)
	}
	for {
		switch nr, er := r.Read(&buf); {
		case nr < 0:
			fmt.Fprintf(os.Stderr, "error reading from %s: %s\n", r.String(), er.String());
			sys.Exit(1);
		case nr == 0:  // EOF
			return;
		case nr > 0:
			nw, ew := file.Stdout.Write(buf[0:nr]);
			if nw != nr {
				fmt.Fprintf(os.Stderr, "error writing from %s: %s\n", r.String(), ew.String());
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
