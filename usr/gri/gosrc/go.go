// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import Build "build"
import Globals "globals"
import Compilation "compilation"


func PrintHelp() {
	print(
		"go (" + Build.time + ")\n" +
		"usage:\n" +
		"  go { flag } { file }\n" +
		"  -d             debug mode, additional self tests and prints\n" +
		"  -o file        explicit object file\n" +
		"  -r             recursively update imported packages in current directory\n" +
		"  -p             print package interface\n" +
		"  -v [0 .. 3]    verbosity level\n" +
		"  -6g            6g compatibility mode\n" +
		"  -token_chan    use token channel to scan and parse in parallel\n"
	);
}


var argno int = 1;
func Next() string {
	arg := "";
	if argno < len(sys.Args) {
		arg = sys.Args[argno];
		argno++;
	}
	return arg;
}


func main() {
	arg := Next();

	if arg == "" {
		PrintHelp();
		return;
	}

	// collect flags and files
	flags := new(Globals.Flags);
	files := Globals.NewList();
	for arg != "" {
	    switch arg {
		case "-d": flags.debug = true;
		case "-o": flags.object_file = Next();
			print("note: -o flag ignored at the moment\n");
		case "-r": flags.update_packages = true;
		case "-p": flags.print_interface = true;
		case "-v":
			arg = Next();
			switch arg {
			case "0", "1", "2", "3":
				flags.verbosity = uint(arg[0] - '0');
			default:
				// anything else is considered the next argument
				flags.verbosity = 1;
				continue;
			}
		case "-6g": flags.sixg = true;
		case "-token_chan": flags.token_chan = true;
		default: files.AddStr(arg);
		}
		arg = Next();
	}

	// setup environment
	env := new(Globals.Environment);
	env.Error = &Compilation.Error;
	env.Import = &Compilation.Import;
	env.Export = &Compilation.Export;
	env.Compile = &Compilation.Compile;

	// compile files
	for p := files.first; p != nil; p = p.next {
		// setup compilation
		comp := new(Globals.Compilation);
		comp.flags = flags;
		comp.env = env;

		// compile
		Compilation.Compile(comp, p.str);
	}
}
