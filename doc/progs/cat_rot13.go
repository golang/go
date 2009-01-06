// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	FD "fd";
	Flag "flag";
)

var rot13_flag = Flag.Bool("rot13", false, nil, "rot13 the input")

func rot13(b byte) byte {
	if 'a' <= b && b <= 'z' {
	   b = 'a' + ((b - 'a') + 13) % 26;
	}
	if 'A' <= b && b <= 'Z' {
	   b = 'A' + ((b - 'A') + 13) % 26
	}
	return b
}

type Reader interface {
	Read(b []byte) (ret int64, errno int64);
	Name() string;
}

type Rot13 struct {
	source	Reader;
}

func NewRot13(source Reader) *Rot13 {
	r13 := new(Rot13);
	r13.source = source;
	return r13
}

func (r13 *Rot13) Read(b []byte) (ret int64, errno int64) {	// TODO: use standard Read sig?
	r, e := r13.source.Read(b);
	for i := int64(0); i < r; i++ {
		b[i] = rot13(b[i])
	}
	return r, e
}

func (r13 *Rot13) Name() string {
	return r13.source.Name()
}
// end of Rot13 implementation

func cat(r Reader) {
	const NBUF = 512;
	var buf [NBUF]byte;

	if rot13_flag.BVal() {
		r = NewRot13(r)
	}
	for {
		switch nr, er := r.Read(buf); {
		case nr < 0:
			print("error reading from ", r.Name(), ": ", er, "\n");
			sys.exit(1);
		case nr == 0:  // EOF
			return;
		case nr > 0:
			nw, ew := FD.Stdout.Write(buf[0:nr]);
			if nw != nr {
				print("error writing from ", r.Name(), ": ", ew, "\n");
			}
		}
	}
}

func main() {
	var bug FD.FD;
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
