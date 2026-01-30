// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements textual dumping of arbitrary data structures
// for debugging purposes. The code is customized for Node graphs
// and may be used for an alternative view of the node structure.

package ir

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/url"
	"os"
	"reflect"
	"regexp"
	"strings"
	"sync"

	"cmd/compile/internal/base"
	"cmd/compile/internal/types"
	"cmd/internal/src"
)

// DumpAny is like FDumpAny but prints to stderr.
func DumpAny(root any, filter string, depth int) {
	FDumpAny(os.Stderr, root, filter, depth)
}

// FDumpAny prints the structure of a rooted data structure
// to w by depth-first traversal of the data structure.
//
// The filter parameter is a regular expression. If it is
// non-empty, only struct fields whose names match filter
// are printed.
//
// The depth parameter controls how deep traversal recurses
// before it returns (higher value means greater depth).
// If an empty field filter is given, a good depth default value
// is 4. A negative depth means no depth limit, which may be fine
// for small data structures or if there is a non-empty filter.
//
// In the output, Node structs are identified by their Op name
// rather than their type; struct fields with zero values or
// non-matching field names are omitted, and "…" means recursion
// depth has been reached or struct fields have been omitted.
func FDumpAny(w io.Writer, root any, filter string, depth int) {
	if root == nil {
		fmt.Fprintln(w, "nil")
		return
	}

	if filter == "" {
		filter = ".*" // default
	}

	p := dumper{
		output:  w,
		fieldrx: regexp.MustCompile(filter),
		ptrmap:  make(map[uintptr]int),
		last:    '\n', // force printing of line number on first line
	}

	p.dump(reflect.ValueOf(root), depth)
	p.printf("\n")
}

// MatchAstDump returns true if the fn matches the value
// of the astdump debug flag.  Fn matches in the following
// cases:
//
//   - astdump == name(fn)
//   - astdump == pkgname(fn).name(fn)
//   - astdump == afterslash(pkgname(fn)).name(fn)
//   - astdump begins with a "~" and what follows "~" is a
//     regular expression matching pkgname(fn).name(fn)
//
// If MatchAstDump returns true, it also prints to os.Stderr
//
//	\nir.Match(<fn>, <astdump>) for <where>\n
func MatchAstDump(fn *Func, where string) bool {
	if len(base.Debug.AstDump) == 0 {
		return false
	}
	return matchForDump(fn, base.Ctxt.Pkgpath, where)
}

var dbgRE *regexp.Regexp
var onceDbgRE sync.Once

func matchForDump(fn *Func, pkgPath, where string) bool {
	dbg := false
	flag := base.Debug.AstDump
	if flag[0] == '~' {
		onceDbgRE.Do(func() { dbgRE = regexp.MustCompile(flag[1:]) })
		dbg = dbgRE.MatchString(pkgPath + "." + FuncName(fn))
	} else {
		dbg = matchPkgFn(pkgPath, FuncName(fn), flag)
	}
	return dbg
}

// matchPkgFn returns true if pkg and fnName "match" toMatch.
// "aFunc" matches "aFunc" (in any package)
// "aPkg.aFunc" matches "aPkg.aFunc"
// "aPkg/subPkg.aFunc" matches "subPkg.aFunc"
func matchPkgFn(pkgName, fnName, toMatch string) bool {
	if fnName == toMatch {
		return true
	}
	matchPkgDotName := func(pkg string) bool {
		// Allocation-free equality check for toMatch == base.Ctxt.Pkgpath + "." + fnName
		return len(toMatch) == len(pkg)+1+len(fnName) &&
			strings.HasPrefix(toMatch, pkg) && toMatch[len(pkg)] == '.' && strings.HasSuffix(toMatch, fnName)
	}
	if matchPkgDotName(pkgName) {
		return true
	}
	if l := strings.LastIndexByte(pkgName, '/'); l > 0 && matchPkgDotName(pkgName[l+1:]) {
		return true
	}

	return false
}

// AstDump appends the ast dump for fn to the ast dump file for fn.
// The generated file name is
//
//	url.PathEscape(PkgFuncName(fn)) + ".ast"
//
// It also prints
//
//	Writing ast output to <astfilename>\n
//
// to os.Stderr.
func AstDump(fn *Func, why string) {
	err := withLockAndFile(
		fn,
		func(w io.Writer) {
			FDump(w, why, fn)
		},
	)
	// strip text following comma, for phase names.
	comma := strings.Index(why, ",")
	if comma > 0 {
		why = why[:comma]
	}
	DumpNodeHTML(fn, why, fn)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Dump returned error %v\n", err)
	}
}

var mu sync.Mutex
var astDumpFiles = make(map[string]bool)

// escapedFileName constructs a file name from fn and suffix,
// url-path-escaping the function part of the name and replacing it
// with a hash if it is too long.  The suffix is neither escaped
// nor including in the length calculation, so an excessively
// creative suffix will result in problems.
func escapedFileName(fn *Func, suffix string) string {
	name := url.PathEscape(PkgFuncName(fn))
	if len(name) > 125 { // arbitrary limit on file names, as if anyone types these in by hand
		hash := sha256.Sum256([]byte(name))
		name = hex.EncodeToString(hash[:8])
	}
	return name + suffix
}

// withLockAndFile manages ast dump files for various function names
// and invokes a dumping function to write output, under a lock.
func withLockAndFile(fn *Func, dump func(io.Writer)) (err error) {
	name := escapedFileName(fn, ".ast")

	// Ensure that debugging output is not scrambled and is written promptly
	mu.Lock()
	defer mu.Unlock()
	mode := os.O_APPEND | os.O_RDWR
	if !astDumpFiles[name] {
		astDumpFiles[name] = true
		mode = os.O_CREATE | os.O_TRUNC | os.O_RDWR
		fmt.Fprintf(os.Stderr, "Writing text ast output for %s to %s\n", PkgFuncName(fn), name)
	}

	fi, err := os.OpenFile(name, mode, 0777)
	if err != nil {
		return err
	}
	defer func() { err = fi.Close() }()
	dump(fi)
	return
}

var htmlWriters = make(map[*Func]*HTMLWriter)
var orderedFuncs = []*Func{}

// DumpNodeHTML dumps the node n to the HTML writer for fn.
// It uses the same phase name as the text dump.
func DumpNodeHTML(fn *Func, why string, n Node) {
	mu.Lock()
	defer mu.Unlock()
	w, ok := htmlWriters[fn]
	if !ok {
		name := escapedFileName(fn, ".html")
		w = NewHTMLWriter(name, fn, "")
		htmlWriters[fn] = w
		orderedFuncs = append(orderedFuncs, fn)
	}
	w.WritePhase(why, why)
}

// CloseHTMLWriter closes the HTML writer for fn, if one exists.
func CloseHTMLWriters() {
	mu.Lock()
	defer mu.Unlock()
	for _, fn := range orderedFuncs {
		if w, ok := htmlWriters[fn]; ok {
			w.Close()
			delete(htmlWriters, fn)
		}
	}
	orderedFuncs = nil
}

type dumper struct {
	output  io.Writer
	fieldrx *regexp.Regexp  // field name filter
	ptrmap  map[uintptr]int // ptr -> dump line number
	lastadr string          // last address string printed (for shortening)

	// output
	indent int  // current indentation level
	last   byte // last byte processed by Write
	line   int  // current line number
}

var indentBytes = []byte(".  ")

func (p *dumper) Write(data []byte) (n int, err error) {
	var m int
	for i, b := range data {
		// invariant: data[0:n] has been written
		if b == '\n' {
			m, err = p.output.Write(data[n : i+1])
			n += m
			if err != nil {
				return
			}
		} else if p.last == '\n' {
			p.line++
			_, err = fmt.Fprintf(p.output, "%6d  ", p.line)
			if err != nil {
				return
			}
			for j := p.indent; j > 0; j-- {
				_, err = p.output.Write(indentBytes)
				if err != nil {
					return
				}
			}
		}
		p.last = b
	}
	if len(data) > n {
		m, err = p.output.Write(data[n:])
		n += m
	}
	return
}

// printf is a convenience wrapper.
func (p *dumper) printf(format string, args ...any) {
	if _, err := fmt.Fprintf(p, format, args...); err != nil {
		panic(err)
	}
}

// addr returns the (hexadecimal) address string of the object
// represented by x (or "?" if x is not addressable), with the
// common prefix between this and the prior address replaced by
// "0x…" to make it easier to visually match addresses.
func (p *dumper) addr(x reflect.Value) string {
	if !x.CanAddr() {
		return "?"
	}
	adr := fmt.Sprintf("%p", x.Addr().Interface())
	s := adr
	if i := commonPrefixLen(p.lastadr, adr); i > 0 {
		s = "0x…" + adr[i:]
	}
	p.lastadr = adr
	return s
}

// dump prints the contents of x.
func (p *dumper) dump(x reflect.Value, depth int) {
	if depth == 0 {
		p.printf("…")
		return
	}

	if pos, ok := x.Interface().(src.XPos); ok {
		p.printf("%s", base.FmtPos(pos))
		return
	}

	switch x.Kind() {
	case reflect.String:
		p.printf("%q", x.Interface()) // print strings in quotes

	case reflect.Interface:
		if x.IsNil() {
			p.printf("nil")
			return
		}
		p.dump(x.Elem(), depth-1)

	case reflect.Ptr:
		if x.IsNil() {
			p.printf("nil")
			return
		}

		p.printf("*")
		ptr := x.Pointer()
		if line, exists := p.ptrmap[ptr]; exists {
			p.printf("(@%d)", line)
			return
		}
		p.ptrmap[ptr] = p.line
		p.dump(x.Elem(), depth) // don't count pointer indirection towards depth

	case reflect.Slice:
		if x.IsNil() {
			p.printf("nil")
			return
		}
		p.printf("%s (%d entries) {", x.Type(), x.Len())
		if x.Len() > 0 {
			p.indent++
			p.printf("\n")
			for i, n := 0, x.Len(); i < n; i++ {
				p.printf("%d: ", i)
				p.dump(x.Index(i), depth-1)
				p.printf("\n")
			}
			p.indent--
		}
		p.printf("}")

	case reflect.Struct:
		typ := x.Type()

		isNode := false
		if n, ok := x.Interface().(Node); ok {
			isNode = true
			p.printf("%s %s {", n.Op().String(), p.addr(x))
		} else {
			p.printf("%s {", typ)
		}
		p.indent++

		first := true
		omitted := false
		for i, n := 0, typ.NumField(); i < n; i++ {
			// Exclude non-exported fields because their
			// values cannot be accessed via reflection.
			if name := typ.Field(i).Name; types.IsExported(name) {
				if !p.fieldrx.MatchString(name) {
					omitted = true
					continue // field name not selected by filter
				}

				// special cases
				if isNode && name == "Op" {
					omitted = true
					continue // Op field already printed for Nodes
				}
				x := x.Field(i)
				if x.IsZero() {
					omitted = true
					continue // exclude zero-valued fields
				}
				if n, ok := x.Interface().(Nodes); ok && len(n) == 0 {
					omitted = true
					continue // exclude empty Nodes slices
				}

				if first {
					p.printf("\n")
					first = false
				}
				p.printf("%s: ", name)
				p.dump(x, depth-1)
				p.printf("\n")
			}
		}
		if omitted {
			p.printf("…\n")
		}

		p.indent--
		p.printf("}")

	default:
		p.printf("%v", x.Interface())
	}
}

func commonPrefixLen(a, b string) (i int) {
	for i < len(a) && i < len(b) && a[i] == b[i] {
		i++
	}
	return
}
