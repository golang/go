// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package xml

import (
	"bufio"
	"io"
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
	MarshalXML() ([]byte, error)
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
// writing nothing.  Marshal handles all other data by writing one or more XML
// elements containing the data.
//
// The name for the XML elements is taken from, in order of preference:
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
// If a field uses a tag "a>b>c", then the element c will be nested inside
// parent elements a and b.  Fields that appear next to each other that name
// the same parent will be enclosed in one XML element.  For example:
//
//	type Result struct {
//		XMLName   xml.Name `xml:"result"`
//		FirstName string   `xml:"person>name>first"`
//		LastName  string   `xml:"person>name>last"`
//		Age       int      `xml:"person>age"`
//	}
//
//	xml.Marshal(w, &Result{FirstName: "John", LastName: "Doe", Age: 42})
//
// would be marshalled as:
//
//	<result>
//		<person>
//			<name>
//				<first>John</first>
//				<last>Doe</last>
//			</name>
//			<age>42</age>
//		</person>
//	</result>
//
// Marshal will return an error if asked to marshal a channel, function, or map.
func Marshal(w io.Writer, v interface{}) (err error) {
	p := &printer{bufio.NewWriter(w)}
	err = p.marshalValue(reflect.ValueOf(v), "???")
	p.Flush()
	return err
}

func (p *printer) marshalValue(val reflect.Value, name string) error {
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
		p.WriteString(strconv.FormatInt(val.Int(), 10))
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		p.WriteString(strconv.FormatUint(val.Uint(), 10))
	case reflect.Float32, reflect.Float64:
		p.WriteString(strconv.FormatFloat(val.Float(), 'g', -1, 64))
	case reflect.String:
		Escape(p, []byte(val.String()))
	case reflect.Bool:
		p.WriteString(strconv.FormatBool(val.Bool()))
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
		s := parentStack{printer: p}
		for i, n := 0, val.NumField(); i < n; i++ {
			if f := typ.Field(i); f.Name != "XMLName" && f.PkgPath == "" {
				name := f.Name
				vf := val.Field(i)
				switch tag := f.Tag.Get("xml"); tag {
				case "":
					s.trim(nil)
				case "chardata":
					if tk := f.Type.Kind(); tk == reflect.String {
						Escape(p, []byte(vf.String()))
					} else if tk == reflect.Slice {
						if elem, ok := vf.Interface().([]byte); ok {
							Escape(p, elem)
						}
					}
					continue
				case "innerxml":
					iface := vf.Interface()
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
					parents := strings.Split(tag, ">")
					if len(parents) == 1 {
						parents, name = nil, tag
					} else {
						parents, name = parents[:len(parents)-1], parents[len(parents)-1]
						if parents[0] == "" {
							parents[0] = f.Name
						}
					}

					s.trim(parents)
					if !(vf.Kind() == reflect.Ptr || vf.Kind() == reflect.Interface) || !vf.IsNil() {
						s.push(parents[len(s.stack):])
					}
				}

				if err := p.marshalValue(vf, name); err != nil {
					return err
				}
			}
		}
		s.trim(nil)
	default:
		return &UnsupportedTypeError{typ}
	}

	p.WriteByte('<')
	p.WriteByte('/')
	p.WriteString(name)
	p.WriteByte('>')

	return nil
}

type parentStack struct {
	*printer
	stack []string
}

// trim updates the XML context to match the longest common prefix of the stack
// and the given parents.  A closing tag will be written for every parent
// popped.  Passing a zero slice or nil will close all the elements.
func (s *parentStack) trim(parents []string) {
	split := 0
	for ; split < len(parents) && split < len(s.stack); split++ {
		if parents[split] != s.stack[split] {
			break
		}
	}

	for i := len(s.stack) - 1; i >= split; i-- {
		s.WriteString("</")
		s.WriteString(s.stack[i])
		s.WriteByte('>')
	}

	s.stack = parents[:split]
}

// push adds parent elements to the stack and writes open tags.
func (s *parentStack) push(parents []string) {
	for i := 0; i < len(parents); i++ {
		s.WriteString("<")
		s.WriteString(parents[i])
		s.WriteByte('>')
	}
	s.stack = append(s.stack, parents...)
}

// A MarshalXMLError is returned when Marshal or MarshalIndent encounter a type
// that cannot be converted into XML.
type UnsupportedTypeError struct {
	Type reflect.Type
}

func (e *UnsupportedTypeError) Error() string {
	return "xml: unsupported type: " + e.Type.String()
}
