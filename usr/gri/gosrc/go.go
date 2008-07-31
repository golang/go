// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import Build "build"
import Globals "globals"
import Compilation "compilation"


// For now we are not using the flags package to minimize
// external dependencies, and because the requirements are
// very minimal at this point.

func PrintHelp() {
  print "go in go (", Build.time, ")\n";
  print "usage:\n";
  print "  go { flag | file }\n";
  print "  -d  print debug information\n";
  print "  -s  enable semantic checks\n";
  print "  -v  verbose mode\n";
  print "  -vv  very verbose mode\n";
}


func main() {
	if sys.argc() <= 1 {
		PrintHelp();
		sys.exit(1);
	}
	
	// collect flags and files
	flags := new(Globals.Flags);
	files := Globals.NewList();
	for i := 1; i < sys.argc(); i++ {
		switch arg := sys.argv(i); arg {
		case "-d": flags.debug = true;
		case "-s": flags.semantic_checks = true;
		case "-v": flags.verbose = 1;
		case "-vv": flags.verbose = 2;
		default: files.AddStr(arg);
		}
	}
	
	// compile files
	for p := files.first; p != nil; p = p.next {
		comp := Globals.NewCompilation(flags);
		Compilation.Compile(comp, p.str);
	}
}
