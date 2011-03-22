// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"flag"
	"fmt"
	"go/parser"
	"go/printer"
	"go/scanner"
	"go/token"
	"io/ioutil"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

var (
	fset     = token.NewFileSet()
	exitCode = 0
)

var allowedRewrites = flag.String("r", "",
	"restrict the rewrites to this comma-separated list")

var allowed map[string]bool

func usage() {
	fmt.Fprintf(os.Stderr, "usage: gofix [-r fixname,...] [path ...]\n")
	flag.PrintDefaults()
	fmt.Fprintf(os.Stderr, "\nAvailable rewrites are:\n")
	for _, f := range fixes {
		fmt.Fprintf(os.Stderr, "\n%s\n", f.name)
		desc := strings.TrimSpace(f.desc)
		desc = strings.Replace(desc, "\n", "\n\t", -1)
		fmt.Fprintf(os.Stderr, "\t%s\n", desc)
	}
	os.Exit(2)
}

func main() {
	sort.Sort(fixes)

	flag.Usage = usage
	flag.Parse()

	if *allowedRewrites != "" {
		allowed = make(map[string]bool)
		for _, f := range strings.Split(*allowedRewrites, ",", -1) {
			allowed[f] = true
		}
	}

	if flag.NArg() == 0 {
		if err := processFile("standard input", true); err != nil {
			report(err)
		}
		os.Exit(exitCode)
	}

	for i := 0; i < flag.NArg(); i++ {
		path := flag.Arg(i)
		switch dir, err := os.Stat(path); {
		case err != nil:
			report(err)
		case dir.IsRegular():
			if err := processFile(path, false); err != nil {
				report(err)
			}
		case dir.IsDirectory():
			walkDir(path)
		}
	}

	os.Exit(exitCode)
}

const (
	tabWidth    = 8
	parserMode  = parser.ParseComments
	printerMode = printer.TabIndent | printer.UseSpaces
)


func processFile(filename string, useStdin bool) os.Error {
	var f *os.File
	var err os.Error

	if useStdin {
		f = os.Stdin
	} else {
		f, err = os.Open(filename, os.O_RDONLY, 0)
		if err != nil {
			return err
		}
		defer f.Close()
	}

	src, err := ioutil.ReadAll(f)
	if err != nil {
		return err
	}

	file, err := parser.ParseFile(fset, filename, src, parserMode)
	if err != nil {
		return err
	}

	fixed := false
	var buf bytes.Buffer
	for _, fix := range fixes {
		if allowed != nil && !allowed[fix.desc] {
			continue
		}
		if fix.f(file) {
			fixed = true
			fmt.Fprintf(&buf, " %s", fix.name)
		}
	}
	if !fixed {
		return nil
	}
	fmt.Fprintf(os.Stderr, "%s: %s\n", filename, buf.String()[1:])

	buf.Reset()
	_, err = (&printer.Config{Mode: printerMode, Tabwidth: tabWidth}).Fprint(&buf, fset, file)
	if err != nil {
		return err
	}

	if useStdin {
		os.Stdout.Write(buf.Bytes())
		return nil
	}

	return ioutil.WriteFile(f.Name(), buf.Bytes(), 0)
}

func report(err os.Error) {
	scanner.PrintError(os.Stderr, err)
	exitCode = 2
}

func walkDir(path string) {
	v := make(fileVisitor)
	go func() {
		filepath.Walk(path, v, v)
		close(v)
	}()
	for err := range v {
		if err != nil {
			report(err)
		}
	}
}

type fileVisitor chan os.Error

func (v fileVisitor) VisitDir(path string, f *os.FileInfo) bool {
	return true
}

func (v fileVisitor) VisitFile(path string, f *os.FileInfo) {
	if isGoFile(f) {
		v <- nil // synchronize error handler
		if err := processFile(path, false); err != nil {
			v <- err
		}
	}
}

func isGoFile(f *os.FileInfo) bool {
	// ignore non-Go files
	return f.IsRegular() && !strings.HasPrefix(f.Name, ".") && strings.HasSuffix(f.Name, ".go")
}
