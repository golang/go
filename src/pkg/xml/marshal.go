// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package xml

import (
	"bufio"
	"io"
	"os"
	"reflect"
	"strconv"
	"strings"
)

const (
	// A generic XML header suitable for use with the output of Marshal and
	// MarshalIndent.  This is not automatically added to any output of this
	// package, it is provided as a convenience.
	Header = `<?xml version="1.0" encoding="UTF-8"?>` + "\n"
)

// A Marshaler can produce well-formatted XML representing its internal state.
// It is used by both Marshal and MarshalIndent.
type Marshaler interface {
	MarshalXML() ([]byte, os.Error)
}

type printer struct {
	*bufio.Writer
}

// Marshal writes an XML-formatted representation of v to w.
//
// If v implements Marshaler, then Marshal calls its MarshalXML method.
// Otherwise, Marshal uses the following procedure to create the XML.
//
// Marshal handles an array or slice by marshalling each of the elements.
// Marshal handles a pointer by marshalling the value it points at or, if the
// pointer is nil, by writing nothing.  Marshal handles an interface value by
// marshalling the value it contains or, if the interface value is nil, by
// writing nothing.  Marshal handles all other data by writing a single XML
// element containing the data.
//
// The name of that XML element is taken from, in order of preference:
//     - the tag on an XMLName field, if the data is a struct
//     - the value of an XMLName field of type xml.Name
//     - the tag of the struct field used to obtain the data
//     - the name of the struct field used to obtain the data
//     - the name '???'.
//
// The XML element for a struct contains marshalled elements for each of the
// exported fields of the struct, with these exceptions:
//     - the XMLName field, described above, is omitted.
//     - a field with tag "attr" becomes an attribute in the XML element.
//     - a field with tag "chardata" is written as character data,
//        not as an XML element.
//     - a field with tag "innerxml" is written verbatim,
//        not subject to the usual marshalling procedure.
//
// Marshal will return an error if asked to marshal a channel, function, or map.
func Marshal(w io.Writer, v interface{}) (err os.Error) {
	p := &printer{bufio.NewWriter(w)}
	err = p.marshalValue(reflect.ValueOf(v), "???")
	p.Flush()
	return err
}

func (p *printer) marshalValue(val reflect.Value, name string) os.Error {
	if !val.IsValid() {
		return nil
	}

	kind := val.Kind()
	typ := val.Type()

	// Try Marshaler
	if typ.NumMethod() > 0 {
		if marshaler, ok := val.Interface().(Marshaler); ok {
			bytes, err := marshaler.MarshalXML()
			if err != nil {
				return err
			}
			p.Write(bytes)
			return nil
		}
	}

	// Drill into pointers/interfaces
	if kind == reflect.Ptr || kind == reflect.Interface {
		if val.IsNil() {
			return nil
		}
		return p.marshalValue(val.Elem(), name)
	}

	// Slices and arrays iterate over the elements. They do not have an enclosing tag.
	if (kind == reflect.Slice || kind == reflect.Array) && typ.Elem().Kind() != reflect.Uint8 {
		for i, n := 0, val.Len(); i < n; i++ {
			if err := p.marshalValue(val.Index(i), name); err != nil {
				return err
			}
		}
		return nil
	}

	// Find XML name
	xmlns := ""
	if kind == reflect.Struct {
		if f, ok := typ.FieldByName("XMLName"); ok {
			if tag := f.Tag.Get("xml"); tag != "" {
				if i := strings.Index(tag, " "); i >= 0 {
					xmlns, name = tag[:i], tag[i+1:]
				} else {
					name = tag
				}
			} else if v, ok := val.FieldByIndex(f.Index).Interface().(Name); ok && v.Local != "" {
				xmlns, name = v.Space, v.Local
			}
		}
	}

	p.WriteByte('<')
	p.WriteString(name)

	// Attributes
	if kind == reflect.Struct {
		if len(xmlns) > 0 {
			p.WriteString(` xmlns="`)
			Escape(p, []byte(xmlns))
			p.WriteByte('"')
		}

		for i, n := 0, typ.NumField(); i < n; i++ {
			if f := typ.Field(i); f.PkgPath == "" && f.Tag.Get("xml") == "attr" {
				if f.Type.Kind() == reflect.String {
					if str := val.Field(i).String(); str != "" {
						p.WriteByte(' ')
						p.WriteString(strings.ToLower(f.Name))
						p.WriteString(`="`)
						Escape(p, []byte(str))
						p.WriteByte('"')
					}
				}
			}
		}
	}
	p.WriteByte('>')

	switch k := val.Kind(); k {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		p.WriteString(strconv.Itoa64(val.Int()))
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		p.WriteString(strconv.Uitoa64(val.Uint()))
	case reflect.Float32, reflect.Float64:
		p.WriteString(strconv.Ftoa64(val.Float(), 'g', -1))
	case reflect.String:
		Escape(p, []byte(val.String()))
	case reflect.Bool:
		p.WriteString(strconv.Btoa(val.Bool()))
	case reflect.Array:
		// will be [...]byte
		bytes := make([]byte, val.Len())
		for i := range bytes {
			bytes[i] = val.Index(i).Interface().(byte)
		}
		Escape(p, bytes)
	case reflect.Slice:
		// will be []byte
		bytes := val.Interface().([]byte)
		Escape(p, bytes)
	case reflect.Struct:
		for i, n := 0, val.NumField(); i < n; i++ {
			if f := typ.Field(i); f.Name != "XMLName" && f.PkgPath == "" {
				name := f.Name
				switch tag := f.Tag.Get("xml"); tag {
				case "":
				case "chardata":
					if tk := f.Type.Kind(); tk == reflect.String {
						Escape(p, []byte(val.Field(i).String()))
					} else if tk == reflect.Slice {
						if elem, ok := val.Field(i).Interface().([]byte); ok {
							Escape(p, elem)
						}
					}
					continue
				case "innerxml":
					iface := val.Field(i).Interface()
					switch raw := iface.(type) {
					case []byte:
						p.Write(raw)
						continue
					case string:
						p.WriteString(raw)
						continue
					}
				case "attr":
					continue
				default:
					name = tag
				}

				if err := p.marshalValue(val.Field(i), name); err != nil {
					return err
				}
			}
		}
	default:
		return &UnsupportedTypeError{typ}
	}

	p.WriteByte('<')
	p.WriteByte('/')
	p.WriteString(name)
	p.WriteByte('>')

	return nil
}

// A MarshalXMLError is returned when Marshal or MarshalIndent encounter a type
// that cannot be converted into XML.
type UnsupportedTypeError struct {
	Type reflect.Type
}

func (e *UnsupportedTypeError) String() string {
	return "xml: unsupported type: " + e.Type.String()
}
