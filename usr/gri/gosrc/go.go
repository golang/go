// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import Build "build"
import Globals "globals"
import Object "object"
import Type "type"
import Universe "universe"
import Scanner "scanner"
import Parser "parser"


func PrintHelp() {
  print "go in go (", Build.time, ")\n";
  print "usage:\n";
  print "  go { -v | -vv | file }\n";
  /*
  printf("flags:\n");
  for (int i = 0; Flags[i].name != NULL; i++) {
    printf("  %s  %s\n", Flags[i].name, Flags[i].help);
  }
  */
}


func Compile(filename, src string, verbose int) {
	Universe.Init();

	S := new(Scanner.Scanner);
	S.Open(filename, src);
	
	P := new(Parser.Parser);
	P.Open(S, verbose);
	
	P.ParseProgram();
}


func main() {
	if sys.argc() <= 1 {
		PrintHelp();
		sys.exit(1);
	}
	
	verbose := 0;
	for i := 1; i < sys.argc(); i++ {
		switch sys.argv(i) {
		case "-v":
			verbose = 1;
			continue;
		case "-vv":
			verbose = 2;
			continue;
		}
		
		src, ok := sys.readfile(sys.argv(i));
		if ok {
			print "parsing " + sys.argv(i) + "\n";
			Compile(sys.argv(i), src, verbose);
		} else {
			print "error: cannot read " + sys.argv(i) + "\n";
		}
	}
}
