// Derived from Plan 9's /sys/src/cmd/units.y
// http://plan9.bell-labs.com/sources/plan9/sys/src/cmd/units.y
//
// Copyright (C) 2003, Lucent Technologies Inc. and others. All Rights Reserved.
// Portions Copyright 2009 The Go Authors.  All Rights Reserved.
// Distributed under the terms of the Lucent Public License Version 1.02
// See http://plan9.bell-labs.com/plan9/license.html

// Generate parser with prefix "units_":
//	go tool yacc -p "units_"

%{

// This tag will end up in the generated y.go, so that forgetting
// 'make clean' does not fail the next build.

// +build ignore

// units.y
// example of a Go yacc program
// usage is
//	go tool yacc -p "units_" units.y (produces y.go)
//	go build -o units y.go
//	./units $GOROOT/src/cmd/yacc/units.txt
//	you have: c
//	you want: furlongs/fortnight
//		* 1.8026178e+12
//		/ 5.5474878e-13
//	you have:

package main

import (
	"bufio"
	"flag"
	"fmt"
	"math"
	"runtime"
	"os"
	"path/filepath"
	"strconv"
	"unicode/utf8"
)

const (
	Ndim = 15  // number of dimensions
	Maxe = 695 // log of largest number
)

type Node struct {
	vval float64
	dim  [Ndim]int8
}

type Var struct {
	name string
	node Node
}

var fi *bufio.Reader // input
var fund [Ndim]*Var  // names of fundamental units
var line string      // current input line
var lineno int       // current input line number
var linep int        // index to next rune in unput
var nerrors int      // error count
var one Node         // constant one
var peekrune rune    // backup runt from input
var retnode1 Node
var retnode2 Node
var retnode Node
var sym string
var vflag bool
%}

%union {
	node Node
	vvar *Var
	numb int
	vval float64
}

%type	<node>	prog expr expr0 expr1 expr2 expr3 expr4

%token	<vval>	VÄL // dieresis to test UTF-8
%token	<vvar>	VAR
%token	<numb>	_SUP // tests leading underscore in token name
%%
prog:
	':' VAR expr
	{
		var f int
		f = int($2.node.dim[0])
		$2.node = $3
		$2.node.dim[0] = 1
		if f != 0 {
			Errorf("redefinition of %v", $2.name)
		} else if vflag {
			fmt.Printf("%v\t%v\n", $2.name, &$2.node)
		}
	}
|	':' VAR '#'
	{
		var f, i int
		for i = 1; i < Ndim; i++ {
			if fund[i] == nil {
				break
			}
		}
		if i >= Ndim {
			Error("too many dimensions")
			i = Ndim - 1
		}
		fund[i] = $2
		f = int($2.node.dim[0])
		$2.node = one
		$2.node.dim[0] = 1
		$2.node.dim[i] = 1
		if f != 0 {
			Errorf("redefinition of %v", $2.name)
		} else if vflag {
			fmt.Printf("%v\t#\n", $2.name)
		}
	}
|	':'
	{
	}
|	'?' expr
	{
		retnode1 = $2
	}
|	'?'
	{
		retnode1 = one
	}

expr:
	expr4
|	expr '+' expr4
	{
		add(&$$, &$1, &$3)
	}
|	expr '-' expr4
	{
		sub(&$$, &$1, &$3)
	}

expr4:
	expr3
|	expr4 '*' expr3
	{
		mul(&$$, &$1, &$3)
	}
|	expr4 '/' expr3
	{
		div(&$$, &$1, &$3)
	}

expr3:
	expr2
|	expr3 expr2
	{
		mul(&$$, &$1, &$2)
	}

expr2:
	expr1
|	expr2 _SUP
	{
		xpn(&$$, &$1, $2)
	}
|	expr2 '^' expr1
	{
		var i int
		for i = 1; i < Ndim; i++ {
			if $3.dim[i] != 0 {
				Error("exponent has units")
				$$ = $1
				break
			}
		}
		if i >= Ndim {
			i = int($3.vval)
			if float64(i) != $3.vval {
				Error("exponent not integral")
			}
			xpn(&$$, &$1, i)
		}
	}

expr1:
	expr0
|	expr1 '|' expr0
	{
		div(&$$, &$1, &$3)
	}

expr0:
	VAR
	{
		if $1.node.dim[0] == 0 {
			Errorf("undefined %v", $1.name)
			$$ = one
		} else {
			$$ = $1.node
		}
	}
|	VÄL
	{
		$$ = one
		$$.vval = $1
	}
|	'(' expr ')'
	{
		$$ = $2
	}
%%

type UnitsLex int

func (UnitsLex) Lex(yylval *units_SymType) int {
	var c rune
	var i int

	c = peekrune
	peekrune = ' '

loop:
	if (c >= '0' && c <= '9') || c == '.' {
		goto numb
	}
	if ralpha(c) {
		goto alpha
	}
	switch c {
	case ' ', '\t':
		c = getrune()
		goto loop
	case '×':
		return '*'
	case '÷':
		return '/'
	case '¹', 'ⁱ':
		yylval.numb = 1
		return _SUP
	case '²', '⁲':
		yylval.numb = 2
		return _SUP
	case '³', '⁳':
		yylval.numb = 3
		return _SUP
	}
	return int(c)

alpha:
	sym = ""
	for i = 0; ; i++ {
		sym += string(c)
		c = getrune()
		if !ralpha(c) {
			break
		}
	}
	peekrune = c
	yylval.vvar = lookup(0)
	return VAR

numb:
	sym = ""
	for i = 0; ; i++ {
		sym += string(c)
		c = getrune()
		if !rdigit(c) {
			break
		}
	}
	peekrune = c
	f, err := strconv.ParseFloat(sym, 64)
	if err != nil {
		fmt.Printf("error converting %v\n", sym)
		f = 0
	}
	yylval.vval = f
	return VÄL
}

func (UnitsLex) Error(s string) {
	Errorf("syntax error, last name: %v", sym)
}

func main() {
	var file string

	flag.BoolVar(&vflag, "v", false, "verbose")

	flag.Parse()

	file = filepath.Join(runtime.GOROOT(), "src/cmd/yacc/units.txt")
	if flag.NArg() > 0 {
		file = flag.Arg(0)
	} else if file == "" {
		fmt.Fprintf(os.Stderr, "cannot find data file units.txt; provide it as argument or set $GOROOT\n")
		os.Exit(1)
	}

	f, err := os.Open(file)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error opening %v: %v\n", file, err)
		os.Exit(1)
	}
	fi = bufio.NewReader(f)

	one.vval = 1

	/*
	 * read the 'units' file to
	 * develop a database
	 */
	lineno = 0
	for {
		lineno++
		if readline() {
			break
		}
		if len(line) == 0 || line[0] == '/' {
			continue
		}
		peekrune = ':'
		units_Parse(UnitsLex(0))
	}

	/*
	 * read the console to
	 * print ratio of pairs
	 */
	fi = bufio.NewReader(os.NewFile(0, "stdin"))

	lineno = 0
	for {
		if (lineno & 1) != 0 {
			fmt.Printf("you want: ")
		} else {
			fmt.Printf("you have: ")
		}
		if readline() {
			break
		}
		peekrune = '?'
		nerrors = 0
		units_Parse(UnitsLex(0))
		if nerrors != 0 {
			continue
		}
		if (lineno & 1) != 0 {
			if specialcase(&retnode, &retnode2, &retnode1) {
				fmt.Printf("\tis %v\n", &retnode)
			} else {
				div(&retnode, &retnode2, &retnode1)
				fmt.Printf("\t* %v\n", &retnode)
				div(&retnode, &retnode1, &retnode2)
				fmt.Printf("\t/ %v\n", &retnode)
			}
		} else {
			retnode2 = retnode1
		}
		lineno++
	}
	fmt.Printf("\n")
	os.Exit(0)
}

/*
 * all characters that have some
 * meaning. rest are usable as names
 */
func ralpha(c rune) bool {
	switch c {
	case 0, '+', '-', '*', '/', '[', ']', '(', ')',
		'^', ':', '?', ' ', '\t', '.', '|', '#',
		'×', '÷', '¹', 'ⁱ', '²', '⁲', '³', '⁳':
		return false
	}
	return true
}

/*
 * number forming character
 */
func rdigit(c rune) bool {
	switch c {
	case '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
		'.', 'e', '+', '-':
		return true
	}
	return false
}

func Errorf(s string, v ...interface{}) {
	fmt.Printf("%v: %v\n\t", lineno, line)
	fmt.Printf(s, v...)
	fmt.Printf("\n")

	nerrors++
	if nerrors > 5 {
		fmt.Printf("too many errors\n")
		os.Exit(1)
	}
}

func Error(s string) {
	Errorf("%s", s)
}

func add(c, a, b *Node) {
	var i int
	var d int8

	for i = 0; i < Ndim; i++ {
		d = a.dim[i]
		c.dim[i] = d
		if d != b.dim[i] {
			Error("add must be like units")
		}
	}
	c.vval = fadd(a.vval, b.vval)
}

func sub(c, a, b *Node) {
	var i int
	var d int8

	for i = 0; i < Ndim; i++ {
		d = a.dim[i]
		c.dim[i] = d
		if d != b.dim[i] {
			Error("sub must be like units")
		}
	}
	c.vval = fadd(a.vval, -b.vval)
}

func mul(c, a, b *Node) {
	var i int

	for i = 0; i < Ndim; i++ {
		c.dim[i] = a.dim[i] + b.dim[i]
	}
	c.vval = fmul(a.vval, b.vval)
}

func div(c, a, b *Node) {
	var i int

	for i = 0; i < Ndim; i++ {
		c.dim[i] = a.dim[i] - b.dim[i]
	}
	c.vval = fdiv(a.vval, b.vval)
}

func xpn(c, a *Node, b int) {
	var i int

	*c = one
	if b < 0 {
		b = -b
		for i = 0; i < b; i++ {
			div(c, c, a)
		}
	} else {
		for i = 0; i < b; i++ {
			mul(c, c, a)
		}
	}
}

func specialcase(c, a, b *Node) bool {
	var i int
	var d, d1, d2 int8

	d1 = 0
	d2 = 0
	for i = 1; i < Ndim; i++ {
		d = a.dim[i]
		if d != 0 {
			if d != 1 || d1 != 0 {
				return false
			}
			d1 = int8(i)
		}
		d = b.dim[i]
		if d != 0 {
			if d != 1 || d2 != 0 {
				return false
			}
			d2 = int8(i)
		}
	}
	if d1 == 0 || d2 == 0 {
		return false
	}

	if fund[d1].name == "°C" && fund[d2].name == "°F" &&
		b.vval == 1 {
		for ll := 0; ll < len(c.dim); ll++ {
			c.dim[ll] = b.dim[ll]
		}
		c.vval = a.vval*9./5. + 32.
		return true
	}

	if fund[d1].name == "°F" && fund[d2].name == "°C" &&
		b.vval == 1 {
		for ll := 0; ll < len(c.dim); ll++ {
			c.dim[ll] = b.dim[ll]
		}
		c.vval = (a.vval - 32.) * 5. / 9.
		return true
	}
	return false
}

func printdim(str string, d, n int) string {
	var v *Var

	if n != 0 {
		v = fund[d]
		if v != nil {
			str += fmt.Sprintf("%v", v.name)
		} else {
			str += fmt.Sprintf("[%d]", d)
		}
		switch n {
		case 1:
			break
		case 2:
			str += "²"
		case 3:
			str += "³"
		default:
			str += fmt.Sprintf("^%d", n)
		}
	}
	return str
}

func (n Node) String() string {
	var str string
	var f, i, d int

	str = fmt.Sprintf("%.7e ", n.vval)

	f = 0
	for i = 1; i < Ndim; i++ {
		d = int(n.dim[i])
		if d > 0 {
			str = printdim(str, i, d)
		} else if d < 0 {
			f = 1
		}
	}

	if f != 0 {
		str += " /"
		for i = 1; i < Ndim; i++ {
			d = int(n.dim[i])
			if d < 0 {
				str = printdim(str, i, -d)
			}
		}
	}

	return str
}

func (v *Var) String() string {
	var str string
	str = fmt.Sprintf("%v %v", v.name, v.node)
	return str
}

func readline() bool {
	s, err := fi.ReadString('\n')
	if err != nil {
		return true
	}
	line = s
	linep = 0
	return false
}

func getrune() rune {
	var c rune
	var n int

	if linep >= len(line) {
		return 0
	}
	c, n = utf8.DecodeRuneInString(line[linep:len(line)])
	linep += n
	if c == '\n' {
		c = 0
	}
	return c
}

var symmap = make(map[string]*Var) // symbol table

func lookup(f int) *Var {
	var p float64
	var w *Var

	v, ok := symmap[sym]
	if ok {
		return v
	}
	if f != 0 {
		return nil
	}
	v = new(Var)
	v.name = sym
	symmap[sym] = v

	p = 1
	for {
		p = fmul(p, pname())
		if p == 0 {
			break
		}
		w = lookup(1)
		if w != nil {
			v.node = w.node
			v.node.vval = fmul(v.node.vval, p)
			break
		}
	}
	return v
}

type Prefix struct {
	vval float64
	name string
}

var prefix = []Prefix{ // prefix table
	{1e-24, "yocto"},
	{1e-21, "zepto"},
	{1e-18, "atto"},
	{1e-15, "femto"},
	{1e-12, "pico"},
	{1e-9, "nano"},
	{1e-6, "micro"},
	{1e-6, "μ"},
	{1e-3, "milli"},
	{1e-2, "centi"},
	{1e-1, "deci"},
	{1e1, "deka"},
	{1e2, "hecta"},
	{1e2, "hecto"},
	{1e3, "kilo"},
	{1e6, "mega"},
	{1e6, "meg"},
	{1e9, "giga"},
	{1e12, "tera"},
	{1e15, "peta"},
	{1e18, "exa"},
	{1e21, "zetta"},
	{1e24, "yotta"},
}

func pname() float64 {
	var i, j, n int
	var s string

	/*
	 * rip off normal prefixs
	 */
	n = len(sym)
	for i = 0; i < len(prefix); i++ {
		s = prefix[i].name
		j = len(s)
		if j < n && sym[0:j] == s {
			sym = sym[j:n]
			return prefix[i].vval
		}
	}

	/*
	 * rip off 's' suffixes
	 */
	if n > 2 && sym[n-1] == 's' {
		sym = sym[0 : n-1]
		return 1
	}

	return 0
}

// careful multiplication
// exponents (log) are checked before multiply
func fmul(a, b float64) float64 {
	var l float64

	if b <= 0 {
		if b == 0 {
			return 0
		}
		l = math.Log(-b)
	} else {
		l = math.Log(b)
	}

	if a <= 0 {
		if a == 0 {
			return 0
		}
		l += math.Log(-a)
	} else {
		l += math.Log(a)
	}

	if l > Maxe {
		Error("overflow in multiply")
		return 1
	}
	if l < -Maxe {
		Error("underflow in multiply")
		return 0
	}
	return a * b
}

// careful division
// exponents (log) are checked before divide
func fdiv(a, b float64) float64 {
	var l float64

	if b <= 0 {
		if b == 0 {
			Errorf("division by zero: %v %v", a, b)
			return 1
		}
		l = math.Log(-b)
	} else {
		l = math.Log(b)
	}

	if a <= 0 {
		if a == 0 {
			return 0
		}
		l -= math.Log(-a)
	} else {
		l -= math.Log(a)
	}

	if l < -Maxe {
		Error("overflow in divide")
		return 1
	}
	if l > Maxe {
		Error("underflow in divide")
		return 0
	}
	return a / b
}

func fadd(a, b float64) float64 {
	return a + b
}
