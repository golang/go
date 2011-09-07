// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"exec"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/scanner"
	"go/token"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"runtime/pprof"
	"strings"
)

var (
	// main operation modes
	list        = flag.Bool("l", false, "list files whose formatting differs from gofmt's")
	write       = flag.Bool("w", false, "write result to (source) file instead of stdout")
	rewriteRule = flag.String("r", "", "rewrite rule (e.g., 'α[β:len(α)] -> α[β:]')")
	simplifyAST = flag.Bool("s", false, "simplify code")
	doDiff      = flag.Bool("d", false, "display diffs instead of rewriting files")
	allErrors   = flag.Bool("e", false, "print all (including spurious) errors")

	// layout control
	comments  = flag.Bool("comments", true, "print comments")
	tabWidth  = flag.Int("tabwidth", 8, "tab width")
	tabIndent = flag.Bool("tabindent", true, "indent with tabs independent of -spaces")
	useSpaces = flag.Bool("spaces", true, "align with spaces instead of tabs")

	// debugging
	cpuprofile = flag.String("cpuprofile", "", "write cpu profile to this file")
)

var (
	fset        = token.NewFileSet()
	exitCode    = 0
	rewrite     func(*ast.File) *ast.File
	parserMode  uint
	printerMode uint
)

func report(err os.Error) {
	scanner.PrintError(os.Stderr, err)
	exitCode = 2
}

func usage() {
	fmt.Fprintf(os.Stderr, "usage: gofmt [flags] [path ...]\n")
	flag.PrintDefaults()
	os.Exit(2)
}

func initParserMode() {
	parserMode = uint(0)
	if *comments {
		parserMode |= parser.ParseComments
	}
	if *allErrors {
		parserMode |= parser.SpuriousErrors
	}
}

func initPrinterMode() {
	printerMode = uint(0)
	if *tabIndent {
		printerMode |= printer.TabIndent
	}
	if *useSpaces {
		printerMode |= printer.UseSpaces
	}
}

func isGoFile(f *os.FileInfo) bool {
	// ignore non-Go files
	return f.IsRegular() && !strings.HasPrefix(f.Name, ".") && strings.HasSuffix(f.Name, ".go")
}

// If in == nil, the source is the contents of the file with the given filename.
func processFile(filename string, in io.Reader, out io.Writer) os.Error {
	if in == nil {
		f, err := os.Open(filename)
		if err != nil {
			return err
		}
		defer f.Close()
		in = f
	}

	src, err := ioutil.ReadAll(in)
	if err != nil {
		return err
	}

	file, err := parser.ParseFile(fset, filename, src, parserMode)
	if err != nil {
		return err
	}

	if rewrite != nil {
		file = rewrite(file)
	}

	if *simplifyAST {
		simplify(file)
	}

	var buf bytes.Buffer
	_, err = (&printer.Config{printerMode, *tabWidth}).Fprint(&buf, fset, file)
	if err != nil {
		return err
	}
	res := buf.Bytes()

	if !bytes.Equal(src, res) {
		// formatting has changed
		if *list {
			fmt.Fprintln(out, filename)
		}
		if *write {
			err = ioutil.WriteFile(filename, res, 0)
			if err != nil {
				return err
			}
		}
		if *doDiff {
			data, err := diff(src, res)
			if err != nil {
				return fmt.Errorf("computing diff: %s", err)
			}
			fmt.Printf("diff %s gofmt/%s\n", filename, filename)
			out.Write(data)
		}
	}

	if !*list && !*write && !*doDiff {
		_, err = out.Write(res)
	}

	return err
}

type fileVisitor chan os.Error

func (v fileVisitor) VisitDir(path string, f *os.FileInfo) bool {
	return true
}

func (v fileVisitor) VisitFile(path string, f *os.FileInfo) {
	if isGoFile(f) {
		v <- nil // synchronize error handler
		if err := processFile(path, nil, os.Stdout); err != nil {
			v <- err
		}
	}
}

func (v fileVisitor) Error(path string, err os.Error) {
	v <- err
}

func walkDir(path string) {
	v := make(fileVisitor)
	go func() {
		filepath.Walk(path, v)
		close(v)
	}()
	for err := range v {
		if err != nil {
			report(err)
		}
	}
}

func main() {
	// call gofmtMain in a separate function
	// so that it can use defer and have them
	// run before the exit.
	gofmtMain()
	os.Exit(exitCode)
}

func gofmtMain() {
	flag.Usage = usage
	flag.Parse()
	if *tabWidth < 0 {
		fmt.Fprintf(os.Stderr, "negative tabwidth %d\n", *tabWidth)
		exitCode = 2
		return
	}

	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "creating cpu profile: %s\n", err)
			exitCode = 2
			return
		}
		defer f.Close()
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	initParserMode()
	initPrinterMode()
	initRewrite()

	if flag.NArg() == 0 {
		if err := processFile("<standard input>", os.Stdin, os.Stdout); err != nil {
			report(err)
		}
		return
	}

	for i := 0; i < flag.NArg(); i++ {
		path := flag.Arg(i)
		switch dir, err := os.Stat(path); {
		case err != nil:
			report(err)
		case dir.IsRegular():
			if err := processFile(path, nil, os.Stdout); err != nil {
				report(err)
			}
		case dir.IsDirectory():
			walkDir(path)
		}
	}
}

func diff(b1, b2 []byte) (data []byte, err os.Error) {
	f1, err := ioutil.TempFile("", "gofmt")
	if err != nil {
		return
	}
	defer os.Remove(f1.Name())
	defer f1.Close()

	f2, err := ioutil.TempFile("", "gofmt")
	if err != nil {
		return
	}
	defer os.Remove(f2.Name())
	defer f2.Close()

	f1.Write(b1)
	f2.Write(b2)

	data, err = exec.Command("diff", "-u", f1.Name(), f2.Name()).CombinedOutput()
	if len(data) > 0 {
		// diff exits with a non-zero status when the files don't match.
		// Ignore that failure as long as we get output.
		err = nil
	}
	return

}
