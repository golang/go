// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"flag"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"log"
	"os"
	"text/template"
)

var (
	srcFn   = flag.String("src", "", "source filename")
	getName = flag.String("name", "", "func/type name to output")
	html    = flag.Bool("html", true, "output HTML")
	showPkg = flag.Bool("pkg", false, "show package in output")
)

func main() {
	// handle input
	flag.Parse()
	if *srcFn == "" || *getName == "" {
		flag.Usage()
		os.Exit(2)
	}
	// load file
	fs := token.NewFileSet()
	file, err := parser.ParseFile(fs, *srcFn, nil, 0)
	if err != nil {
		log.Fatal(err)
	}
	// create filter
	filter := func(name string) bool {
		return name == *getName
	}
	// filter
	if !ast.FilterFile(file, filter) {
		os.Exit(1)
	}
	// print the AST
	var b bytes.Buffer
	printer.Fprint(&b, fs, file)
	// drop package declaration
	if !*showPkg {
		for {
			c, err := b.ReadByte()
			if c == '\n' || err != nil {
				break
			}
		}
	}
	// drop leading newlines
	for {
		b, err := b.ReadByte()
		if err != nil {
			break
		}
		if b != '\n' {
			os.Stdout.Write([]byte{b})
			break
		}
	}
	// output
	if *html {
		template.HTMLEscape(os.Stdout, b.Bytes())
	} else {
		b.WriteTo(os.Stdout)
	}
}
