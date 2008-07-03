// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import Scanner "scanner"


func Scan(src string) {
	S := new(Scanner.Scanner);
	S.Open(src);
	for {
		var tok, beg, end int;
		tok, beg, end = S.Scan();
		print Scanner.TokenName(tok), "\t ", src[beg : end], "\n";
		if tok == Scanner.EOF {
			return;
		}
	}
}


func main() {
	for i := 1; i < sys.argc(); i++ {
		var src string;
		var ok bool;
		src, ok = sys.readfile(sys.argv(i));
		if ok {
			print "scanning " + sys.argv(i) + "\n";
			Scan(src);
		} else {
			print "error: cannot read " + sys.argv(i) + "\n";
		}
	}
}
