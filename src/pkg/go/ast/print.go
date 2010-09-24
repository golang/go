// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains printing suppport for ASTs.

package ast

import (
	"fmt"
	"go/token"
	"io"
	"os"
	"reflect"
)


// A FieldFilter may be provided to Fprint to control the output.
type FieldFilter func(name string, value reflect.Value) bool


// NotNilFilter returns true for field values that are not nil;
// it returns false otherwise.
func NotNilFilter(_ string, value reflect.Value) bool {
	v, ok := value.(interface {
		IsNil() bool
	})
	return !ok || !v.IsNil()
}


// Fprint prints the (sub-)tree starting at AST node x to w.
//
// A non-nil FieldFilter f may be provided to control the output:
// struct fields for which f(fieldname, fieldvalue) is true are
// are printed; all others are filtered from the output.
//
func Fprint(w io.Writer, x interface{}, f FieldFilter) (n int, err os.Error) {
	// setup printer
	p := printer{
		output: w,
		filter: f,
		ptrmap: make(map[interface{}]int),
		last:   '\n', // force printing of line number on first line
	}

	// install error handler
	defer func() {
		n = p.written
		if e := recover(); e != nil {
			err = e.(localError).err // re-panics if it's not a localError
		}
	}()

	// print x
	if x == nil {
		p.printf("nil\n")
		return
	}
	p.print(reflect.NewValue(x))
	p.printf("\n")

	return
}


// Print prints x to standard output, skipping nil fields.
// Print(x) is the same as Fprint(os.Stdout, x, NotNilFilter).
func Print(x interface{}) (int, os.Error) {
	return Fprint(os.Stdout, x, NotNilFilter)
}


type printer struct {
	output  io.Writer
	filter  FieldFilter
	ptrmap  map[interface{}]int // *reflect.PtrValue -> line number
	written int                 // number of bytes written to output
	indent  int                 // current indentation level
	last    byte                // the last byte processed by Write
	line    int                 // current line number
}


var indent = []byte(".  ")

func (p *printer) Write(data []byte) (n int, err os.Error) {
	var m int
	for i, b := range data {
		// invariant: data[0:n] has been written
		if b == '\n' {
			m, err = p.output.Write(data[n : i+1])
			n += m
			if err != nil {
				return
			}
			p.line++
		} else if p.last == '\n' {
			_, err = fmt.Fprintf(p.output, "%6d  ", p.line)
			if err != nil {
				return
			}
			for j := p.indent; j > 0; j-- {
				_, err = p.output.Write(indent)
				if err != nil {
					return
				}
			}
		}
		p.last = b
	}
	m, err = p.output.Write(data[n:])
	n += m
	return
}


// localError wraps locally caught os.Errors so we can distinguish
// them from genuine panics which we don't want to return as errors.
type localError struct {
	err os.Error
}


// printf is a convenience wrapper that takes care of print errors.
func (p *printer) printf(format string, args ...interface{}) {
	n, err := fmt.Fprintf(p, format, args...)
	p.written += n
	if err != nil {
		panic(localError{err})
	}
}


// Implementation note: Print is written for AST nodes but could be
// used to print arbitrary data structures; such a version should
// probably be in a different package.

func (p *printer) print(x reflect.Value) {
	// Note: This test is only needed because AST nodes
	//       embed a token.Position, and thus all of them
	//       understand the String() method (but it only
	//       applies to the Position field).
	// TODO: Should reconsider this AST design decision.
	if pos, ok := x.Interface().(token.Position); ok {
		p.printf("%s", pos)
		return
	}

	if !NotNilFilter("", x) {
		p.printf("nil")
		return
	}

	switch v := x.(type) {
	case *reflect.InterfaceValue:
		p.print(v.Elem())

	case *reflect.MapValue:
		p.printf("%s (len = %d) {\n", x.Type().String(), v.Len())
		p.indent++
		for _, key := range v.Keys() {
			p.print(key)
			p.printf(": ")
			p.print(v.Elem(key))
		}
		p.indent--
		p.printf("}")

	case *reflect.PtrValue:
		p.printf("*")
		// type-checked ASTs may contain cycles - use ptrmap
		// to keep track of objects that have been printed
		// already and print the respective line number instead
		ptr := v.Interface()
		if line, exists := p.ptrmap[ptr]; exists {
			p.printf("(obj @ %d)", line)
		} else {
			p.ptrmap[ptr] = p.line
			p.print(v.Elem())
		}

	case *reflect.SliceValue:
		if s, ok := v.Interface().([]byte); ok {
			p.printf("%#q", s)
			return
		}
		p.printf("%s (len = %d) {\n", x.Type().String(), v.Len())
		p.indent++
		for i, n := 0, v.Len(); i < n; i++ {
			p.printf("%d: ", i)
			p.print(v.Elem(i))
			p.printf("\n")
		}
		p.indent--
		p.printf("}")

	case *reflect.StructValue:
		p.printf("%s {\n", x.Type().String())
		p.indent++
		t := v.Type().(*reflect.StructType)
		for i, n := 0, t.NumField(); i < n; i++ {
			name := t.Field(i).Name
			value := v.Field(i)
			if p.filter == nil || p.filter(name, value) {
				p.printf("%s: ", name)
				p.print(value)
				p.printf("\n")
			}
		}
		p.indent--
		p.printf("}")

	default:
		p.printf("%v", x.Interface())
	}
}
