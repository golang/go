// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"cmd/asm/internal/arch"
	"cmd/asm/internal/asm"
	"cmd/asm/internal/flags"
	"cmd/asm/internal/lex"

	"cmd/internal/obj"
)

func main() {
	log.SetFlags(0)
	log.SetPrefix("asm: ")

	GOARCH := obj.Getgoarch()

	architecture := arch.Set(GOARCH)
	if architecture == nil {
		log.Fatalf("asm: unrecognized architecture %s", GOARCH)
	}

	flags.Parse()

	// Create object file, write header.
	fd, err := os.Create(*flags.OutputFile)
	if err != nil {
		log.Fatal(err)
	}
	ctxt := obj.Linknew(architecture.LinkArch)
	if *flags.PrintOut {
		ctxt.Debugasm = 1
	}
	ctxt.LineHist.TrimPathPrefix = *flags.TrimPath
	ctxt.Flag_dynlink = *flags.Dynlink
	if *flags.Shared || *flags.Dynlink {
		ctxt.Flag_shared = 1
	}
	ctxt.Bso = obj.Binitw(os.Stdout)
	defer ctxt.Bso.Flush()
	output := obj.Binitw(fd)
	fmt.Fprintf(output, "go object %s %s %s\n", obj.Getgoos(), obj.Getgoarch(), obj.Getgoversion())
	fmt.Fprintf(output, "!\n")

	lexer := lex.NewLexer(flag.Arg(0), ctxt)
	parser := asm.NewParser(ctxt, architecture, lexer)
	diag := false
	ctxt.DiagFunc = func(format string, args ...interface{}) {
		diag = true
		log.Printf(format, args...)
	}
	pList := obj.Linknewplist(ctxt)
	var ok bool
	pList.Firstpc, ok = parser.Parse()
	if ok {
		// reports errors to parser.Errorf
		obj.Writeobjdirect(ctxt, output)
	}
	if !ok || diag {
		log.Printf("asm: assembly of %s failed", flag.Arg(0))
		os.Remove(*flags.OutputFile)
		os.Exit(1)
	}
	output.Flush()
}
