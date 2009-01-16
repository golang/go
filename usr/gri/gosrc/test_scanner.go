// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import Scanner "scanner"


func Scan1(filename, src string) {
	S := new(Scanner.Scanner);
	S.Open(filename, src);
	for {
		tok, pos, val := S.Scan();
		print(pos, ": ", Scanner.TokenName(tok));
		if tok == Scanner.IDENT || tok == Scanner.INT || tok == Scanner.FLOAT || tok == Scanner.STRING {
			print(" ", val);
		}
		print("\n");
		if tok == Scanner.EOF {
			return;
		}
	}
}


func Scan2(filename, src string) {
	S := new(Scanner.Scanner);
	S.Open(filename, src);
	c := make(chan *Scanner.Token, 32);
	go S.Server(c);
	for {
		var t *Scanner.Token;
		t = <- c;
		tok, pos, val := t.tok, t.pos, t.val;
		print(pos, ": ", Scanner.TokenName(tok));
		if tok == Scanner.IDENT || tok == Scanner.INT || tok == Scanner.FLOAT || tok == Scanner.STRING {
			print(" ", val);
		}
		print("\n");
		if tok == Scanner.EOF {
			return;
		}
	}
}


func main() {
	for i := 1; i < len(sys.Args); i++ {
		var src string;
		var ok bool;
		src, ok = sys.readfile(sys.Args[i]);
		if ok {
			print("scanning (standard) " + sys.Args[i] + "\n");
			Scan1(sys.Args[i], src);
			print("\n");
			print("scanning (channels) " + sys.Args[i] + "\n");
			Scan2(sys.Args[i], src);
		} else {
			print("error: cannot read " + sys.Args[i] + "\n");
		}
		print("\n");
	}
}
