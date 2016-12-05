// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"

	"cmd/asm/internal/arch"
	"cmd/asm/internal/asm"
	"cmd/asm/internal/flags"
	"cmd/asm/internal/lex"

	"cmd/internal/bio"
	"cmd/internal/obj"
)

func main() {
	log.SetFlags(0)
	log.SetPrefix("asm: ")

	GOARCH := obj.GOARCH

	architecture := arch.Set(GOARCH)
	if architecture == nil {
		log.Fatalf("unrecognized architecture %s", GOARCH)
	}

	flags.Parse()

	ctxt := obj.Linknew(architecture.LinkArch)
	if *flags.PrintOut {
		ctxt.Debugasm = 1
	}
	ctxt.LineHist.TrimPathPrefix = *flags.TrimPath
	ctxt.Flag_dynlink = *flags.Dynlink
	ctxt.Flag_shared = *flags.Shared || *flags.Dynlink
	ctxt.Bso = bufio.NewWriter(os.Stdout)
	defer ctxt.Bso.Flush()

	// Create object file, write header.
	out, err := os.Create(*flags.OutputFile)
	if err != nil {
		log.Fatal(err)
	}
	defer bio.MustClose(out)
	buf := bufio.NewWriter(bio.MustWriter(out))

	fmt.Fprintf(buf, "go object %s %s %s\n", obj.GOOS, obj.GOARCH, obj.Version)
	fmt.Fprintf(buf, "!\n")

	var ok, diag bool
	var failedFile string
	for _, f := range flag.Args() {
		lexer := lex.NewLexer(f, ctxt)
		parser := asm.NewParser(ctxt, architecture, lexer)
		ctxt.DiagFunc = func(format string, args ...interface{}) {
			diag = true
			log.Printf(format, args...)
		}
		pList := obj.Linknewplist(ctxt)
		pList.Firstpc, ok = parser.Parse()
		if !ok {
			failedFile = f
			break
		}
	}
	if ok {
		// reports errors to parser.Errorf
		obj.Writeobjdirect(ctxt, buf)
	}
	if !ok || diag {
		if failedFile != "" {
			log.Printf("assembly of %s failed", failedFile)
		} else {
			log.Print("assembly failed")
		}
		out.Close()
		os.Remove(*flags.OutputFile)
		os.Exit(1)
	}
	buf.Flush()
}
