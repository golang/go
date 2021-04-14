// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements printing of syntax tree structures.

package syntax

import (
	"fmt"
	"io"
	"reflect"
	"unicode"
	"unicode/utf8"
)

// Fdump dumps the structure of the syntax tree rooted at n to w.
// It is intended for debugging purposes; no specific output format
// is guaranteed.
func Fdump(w io.Writer, n Node) (err error) {
	p := dumper{
		output: w,
		ptrmap: make(map[Node]int),
		last:   '\n', // force printing of line number on first line
	}

	defer func() {
		if e := recover(); e != nil {
			err = e.(localError).err // re-panics if it's not a localError
		}
	}()

	if n == nil {
		p.printf("nil\n")
		return
	}
	p.dump(reflect.ValueOf(n), n)
	p.printf("\n")

	return
}

type dumper struct {
	output io.Writer
	ptrmap map[Node]int // node -> dump line number
	indent int          // current indentation level
	last   byte         // last byte processed by Write
	line   int          // current line number
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

// localError wraps locally caught errors so we can distinguish
// them from genuine panics which we don't want to return as errors.
type localError struct {
	err error
}

// printf is a convenience wrapper that takes care of print errors.
func (p *dumper) printf(format string, args ...interface{}) {
	if _, err := fmt.Fprintf(p, format, args...); err != nil {
		panic(localError{err})
	}
}

// dump prints the contents of x.
// If x is the reflect.Value of a struct s, where &s
// implements Node, then &s should be passed for n -
// this permits printing of the unexported span and
// comments fields of the embedded isNode field by
// calling the Span() and Comment() instead of using
// reflection.
func (p *dumper) dump(x reflect.Value, n Node) {
	switch x.Kind() {
	case reflect.Interface:
		if x.IsNil() {
			p.printf("nil")
			return
		}
		p.dump(x.Elem(), nil)

	case reflect.Ptr:
		if x.IsNil() {
			p.printf("nil")
			return
		}

		// special cases for identifiers w/o attached comments (common case)
		if x, ok := x.Interface().(*Name); ok {
			p.printf("%s @ %v", x.Value, x.Pos())
			return
		}

		p.printf("*")
		// Fields may share type expressions, and declarations
		// may share the same group - use ptrmap to keep track
		// of nodes that have been printed already.
		if ptr, ok := x.Interface().(Node); ok {
			if line, exists := p.ptrmap[ptr]; exists {
				p.printf("(Node @ %d)", line)
				return
			}
			p.ptrmap[ptr] = p.line
			n = ptr
		}
		p.dump(x.Elem(), n)

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
				p.dump(x.Index(i), nil)
				p.printf("\n")
			}
			p.indent--
		}
		p.printf("}")

	case reflect.Struct:
		typ := x.Type()

		// if span, ok := x.Interface().(lexical.Span); ok {
		// 	p.printf("%s", &span)
		// 	return
		// }

		p.printf("%s {", typ)
		p.indent++

		first := true
		if n != nil {
			p.printf("\n")
			first = false
			// p.printf("Span: %s\n", n.Span())
			// if c := *n.Comments(); c != nil {
			// 	p.printf("Comments: ")
			// 	p.dump(reflect.ValueOf(c), nil) // a Comment is not a Node
			// 	p.printf("\n")
			// }
		}

		for i, n := 0, typ.NumField(); i < n; i++ {
			// Exclude non-exported fields because their
			// values cannot be accessed via reflection.
			if name := typ.Field(i).Name; isExported(name) {
				if first {
					p.printf("\n")
					first = false
				}
				p.printf("%s: ", name)
				p.dump(x.Field(i), nil)
				p.printf("\n")
			}
		}

		p.indent--
		p.printf("}")

	default:
		switch x := x.Interface().(type) {
		case string:
			// print strings in quotes
			p.printf("%q", x)
		default:
			p.printf("%v", x)
		}
	}
}

func isExported(name string) bool {
	ch, _ := utf8.DecodeRuneInString(name)
	return unicode.IsUpper(ch)
}
