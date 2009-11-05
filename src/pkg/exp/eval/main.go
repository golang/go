// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"./_obj/eval";
	"bufio";
	"flag";
	"go/parser";
	"go/scanner";
	"io";
	"os";
)

var filename = flag.String("f", "", "file to run")

func main() {
	flag.Parse();
	w := eval.NewWorld();
	if *filename != "" {
		data, err := io.ReadFile(*filename);
		if err != nil {
			println(err.String());
			os.Exit(1);
		}
		file, err := parser.ParseFile(*filename, data, 0);
		if err != nil {
			println(err.String());
			os.Exit(1);
		}
		code, err := w.CompileDeclList(file.Decls);
		if err != nil {
			if list, ok := err.(scanner.ErrorList); ok {
				for _, e := range list {
					println(e.String());
				}
			} else {
				println(err.String());
			}
			os.Exit(1);
		}
		_, err := code.Run();
		if err != nil {
			println(err.String());
			os.Exit(1);
		}
		code, err = w.Compile("init()");
		if code != nil {
			_, err := code.Run();
			if err != nil {
				println(err.String());
				os.Exit(1);
			}
		}
		code, err = w.Compile("main()");
		if err != nil {
			println(err.String());
			os.Exit(1);
		}
		_, err = code.Run();
		if err != nil {
			println(err.String());
			os.Exit(1);
		}
		os.Exit(0);
	}

	r := bufio.NewReader(os.Stdin);
	for {
		print("; ");
		line, err := r.ReadString('\n');
		if err != nil {
			break;
		}
		code, err := w.Compile(line);
		if err != nil {
			println(err.String());
			continue;
		}
		v, err := code.Run();
		if err != nil {
			println(err.String());
			continue;
		}
		if v != nil {
			println(v.String());
		}
	}
}
