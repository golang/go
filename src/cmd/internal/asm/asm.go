// Inferno utils/6a/a.h and lex.c.
// http://code.google.com/p/inferno-os/source/browse/utils/6a/a.h
// http://code.google.com/p/inferno-os/source/browse/utils/6a/lex.c
//
//	Copyright © 1994-1999 Lucent Technologies Inc.	All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors.  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// Package asm holds code shared among the assemblers.
package asm

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"cmd/internal/obj"
)

// Initialized by client.
var (
	LSCONST int
	LCONST  int
	LFCONST int
	LNAME   int
	LVAR    int
	LLAB    int

	Thechar     rune
	Thestring   string
	Thelinkarch *obj.LinkArch

	Arches map[string]*obj.LinkArch

	Cclean  func()
	Yyparse func()
	Syminit func(*Sym)

	Lexinit []Lextab
)

type Lextab struct {
	Name  string
	Type  int
	Value int64
}

const (
	MAXALIGN = 7
	FPCHIP   = 1
	NSYMB    = 500
	BUFSIZ   = 8192
	HISTSZ   = 20
	EOF      = -1
	IGN      = -2
	NHASH    = 503
	STRINGSZ = 200
	NMACRO   = 10
)

const (
	CLAST = iota
	CMACARG
	CMACRO
	CPREPROC
)

type Macro struct {
	Text string
	Narg int
	Dots bool
}

type Sym struct {
	Link      *Sym
	Ref       *Ref
	Macro     *Macro
	Value     int64
	Type      int
	Name      string
	Labelname string
	Sym       int8
}

type Ref struct {
	Class int
}

type Io struct {
	Link *Io
	P    []byte
	F    *os.File
	B    [1024]byte
}

var fi struct {
	P []byte
}

var (
	debug    [256]int
	hash     = map[string]*Sym{}
	Dlist    []string
	newflag  int
	hunk     string
	include  []string
	iofree   *Io
	ionext   *Io
	iostack  *Io
	Lineno   int32
	nerrors  int
	nhunk    int32
	ninclude int
	nsymb    int32
	nullgen  obj.Addr
	outfile  string
	Pass     int
	PC       int32
	peekc    int = IGN
	sym      int
	symb     string
	thunk    int32
	obuf     obj.Biobuf
	Ctxt     *obj.Link
	bstdout  obj.Biobuf
)

func dodef(p string) {
	Dlist = append(Dlist, p)
}

func usage() {
	fmt.Printf("usage: %ca [options] file.c...\n", Thechar)
	flag.PrintDefaults()
	errorexit()
}

func Main() {
	var p string

	// Allow GOARCH=Thestring or GOARCH=Thestringsuffix,
	// but not other values.
	p = obj.Getgoarch()

	if !strings.HasPrefix(p, Thestring) {
		log.Fatalf("cannot use %cc with GOARCH=%s", Thechar, p)
	}
	if p != Thestring {
		Thelinkarch = Arches[p]
		if Thelinkarch == nil {
			log.Fatalf("unknown arch %s", p)
		}
	}

	Ctxt = obj.Linknew(Thelinkarch)
	Ctxt.Diag = Yyerror
	Ctxt.Bso = &bstdout
	Ctxt.Enforce_data_order = 1
	bstdout = *obj.Binitw(os.Stdout)

	debug = [256]int{}
	cinit()
	outfile = ""
	setinclude(".")

	flag.Var(flagFn(dodef), "D", "name[=value]: add #define")
	flag.Var(flagFn(setinclude), "I", "dir: add dir to include path")
	flag.Var((*count)(&debug['S']), "S", "print assembly and machine code")
	flag.Var((*count)(&debug['m']), "m", "debug preprocessor macros")
	flag.StringVar(&outfile, "o", "", "file: set output file")
	flag.StringVar(&Ctxt.Trimpath, "trimpath", "", "prefix: remove prefix from recorded source file paths")

	flag.Parse()

	Ctxt.Debugasm = int32(debug['S'])

	if flag.NArg() < 1 {
		usage()
	}
	if flag.NArg() > 1 {
		fmt.Printf("can't assemble multiple files\n")
		errorexit()
	}

	if assemble(flag.Arg(0)) != 0 {
		errorexit()
	}
	obj.Bflush(&bstdout)
	if nerrors > 0 {
		errorexit()
	}
}

func assemble(file string) int {
	var i int

	if outfile == "" {
		outfile = strings.TrimSuffix(filepath.Base(file), ".s") + "." + string(Thechar)
	}

	of, err := os.Create(outfile)
	if err != nil {
		Yyerror("%ca: cannot create %s", Thechar, outfile)
		errorexit()
	}

	obuf = *obj.Binitw(of)
	fmt.Fprintf(&obuf, "go object %s %s %s\n", obj.Getgoos(), obj.Getgoarch(), obj.Getgoversion())
	fmt.Fprintf(&obuf, "!\n")

	for Pass = 1; Pass <= 2; Pass++ {
		pinit(file)
		for i = 0; i < len(Dlist); i++ {
			dodefine(Dlist[i])
		}
		Yyparse()
		Cclean()
		if nerrors != 0 {
			return nerrors
		}
	}

	obj.Writeobjdirect(Ctxt, &obuf)
	obj.Bflush(&obuf)
	return 0
}

func cinit() {
	for i := 0; i < len(Lexinit); i++ {
		s := Lookup(Lexinit[i].Name)
		if s.Type != LNAME {
			Yyerror("double initialization %s", Lexinit[i].Name)
		}
		s.Type = Lexinit[i].Type
		s.Value = Lexinit[i].Value
	}
}

func syminit(s *Sym) {
	s.Type = LNAME
	s.Value = 0
}

type flagFn func(string)

func (flagFn) String() string {
	return "<arg>"
}

func (f flagFn) Set(s string) error {
	f(s)
	return nil
}

type yyImpl struct{}

// count is a flag.Value that is like a flag.Bool and a flag.Int.
// If used as -name, it increments the count, but -name=x sets the count.
// Used for verbose flag -v.
type count int

func (c *count) String() string {
	return fmt.Sprint(int(*c))
}

func (c *count) Set(s string) error {
	switch s {
	case "true":
		*c++
	case "false":
		*c = 0
	default:
		n, err := strconv.Atoi(s)
		if err != nil {
			return fmt.Errorf("invalid count %q", s)
		}
		*c = count(n)
	}
	return nil
}

func (c *count) IsBoolFlag() bool {
	return true
}
