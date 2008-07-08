// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import Scanner "scanner"
import Parser "parser"


func Parse(src string, verbose bool) {
	S := new(Scanner.Scanner);
	S.Open(src);
	
	P := new(Parser.Parser);
	P.Open(S, verbose);
	
	P.ParseProgram();
}


func main() {
	verbose := false;
	for i := 1; i < sys.argc(); i++ {
		if sys.argv(i) == "-v" {
			verbose = true;
			continue;
		}
		
		var src string;
		var ok bool;
		src, ok = sys.readfile(sys.argv(i));
		if ok {
			print "parsing " + sys.argv(i) + "\n";
			Parse(src, verbose);
		} else {
			print "error: cannot read " + sys.argv(i) + "\n";
		}
	}
}
