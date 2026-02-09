// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"reflect"
	"strconv"
)

func pprints(v any) string {
	var pp pprinter
	pp.val(reflect.ValueOf(v), 0)
	return string(pp.buf)
}

type pprinter struct {
	buf []byte
}

func (p *pprinter) indent(by int) {
	for range by {
		p.buf = append(p.buf, '\t')
	}
}

func (p *pprinter) val(v reflect.Value, indent int) {
	switch v.Kind() {
	default:
		p.buf = fmt.Appendf(p.buf, "unsupported kind %v", v.Kind())

	case reflect.Bool:
		p.buf = strconv.AppendBool(p.buf, v.Bool())

	case reflect.Int, reflect.Int16, reflect.Int32, reflect.Int64:
		p.buf = strconv.AppendInt(p.buf, v.Int(), 10)

	case reflect.String:
		p.buf = strconv.AppendQuote(p.buf, v.String())

	case reflect.Pointer:
		if v.IsNil() {
			p.buf = append(p.buf, "nil"...)
		} else {
			p.buf = append(p.buf, "&"...)
			p.val(v.Elem(), indent)
		}

	case reflect.Slice, reflect.Array:
		p.buf = append(p.buf, "[\n"...)
		for i := range v.Len() {
			p.indent(indent + 1)
			p.val(v.Index(i), indent+1)
			p.buf = append(p.buf, ",\n"...)
		}
		p.indent(indent)
		p.buf = append(p.buf, ']')

	case reflect.Struct:
		vt := v.Type()
		p.buf = append(append(p.buf, vt.String()...), "{\n"...)
		for f := range v.NumField() {
			p.indent(indent + 1)
			p.buf = append(append(p.buf, vt.Field(f).Name...), ": "...)
			p.val(v.Field(f), indent+1)
			p.buf = append(p.buf, ",\n"...)
		}
		p.indent(indent)
		p.buf = append(p.buf, '}')
	}
}
