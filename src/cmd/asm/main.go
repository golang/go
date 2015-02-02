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

	flags.Parse(architecture.Thechar)

	// Create object file, write header.
	fd, err := os.Create(*flags.OutputFile)
	if err != nil {
		log.Fatal(err)
	}
	ctxt := obj.Linknew(architecture.LinkArch)
	if *flags.PrintOut {
		ctxt.Debugasm = 1
	}
	ctxt.Bso = obj.Binitw(os.Stdout)
	defer obj.Bflush(ctxt.Bso)
	ctxt.Diag = log.Fatalf
	output := obj.Binitw(fd)
	fmt.Fprintf(output, "go object %s %s %s\n", obj.Getgoos(), obj.Getgoarch(), obj.Getgoversion())
	fmt.Fprintf(output, "!\n")

	lexer := lex.NewLexer(flag.Arg(0), ctxt)
	parser := asm.NewParser(ctxt, architecture, lexer)
	pList := obj.Linknewplist(ctxt)
	var ok bool
	pList.Firstpc, ok = parser.Parse()
	if !ok {
		log.Fatalf("asm: assembly of %s failed", flag.Arg(0))
		os.Exit(1)
	}
	obj.Writeobjdirect(ctxt, output)
	obj.Bflush(output)
}
