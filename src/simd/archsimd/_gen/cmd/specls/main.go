// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// specls prints the expanded API from the SIMD spec package.
package main

import (
	"flag"
	"fmt"
	"go/ast"
	"log"
	"os"
	"regexp"
	"simd/archsimd/_gen/specgen"
)

var (
	flagFilter = flag.String("f", "", "only process spec functions matching `regexp`")
	flagTrace  = flag.Bool("trace", false, "trace solver steps")
)

func main() {
	flag.Usage = func() {
		w := flag.CommandLine.Output()
		fmt.Fprintf(w, "usage: specls [flags] [spec dir]\n")
		flag.CommandLine.PrintDefaults()
	}

	flag.Parse()
	var specDir string
	switch flag.NArg() {
	case 0:
		specDir = specgen.MustFindSpecDir()
	case 1:
		specDir = flag.Arg(0)
	default:
		flag.Usage()
		os.Exit(1)
	}

	var opts specgen.LoadOptions
	if *flagFilter != "" {
		re, err := regexp.Compile(*flagFilter)
		if err != nil {
			log.Fatal("invalid -f: ", err)
		}
		opts.Filter = func(fd *ast.FuncDecl) bool {
			return re.MatchString(fd.Name.Name)
		}
	}
	if *flagTrace {
		opts.Trace = os.Stderr
	}

	funcs, err := specgen.Load(specDir, &opts)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s\n\n", err.Error())
		defer os.Exit(1)
	}

	for _, fn := range funcs {
		fmt.Print(fn.Decl())
		fmt.Println()
	}
}
