// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package xml

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"reflect"
	"strconv"
	"strings"
)

const (
	// A generic XML header suitable for use with the output of Marshal.
	// This is not automatically added to any output of this package,
	// it is provided as a convenience.
	Header = `<?xml version="1.0" encoding="UTF-8"?>` + "\n"
)

// Marshal returns the XML encoding of v.
//
// Marshal handles an array or slice by marshalling each of the elements.
// Marshal handles a pointer by marshalling the value it points at or, if the
// pointer is nil, by writing nothing.  Marshal handles an interface value by
// marshalling the value it contains or, if the interface value is nil, by
// writing nothing.  Marshal handles all other data by writing one or more XML
// elements containing the data.
//
// The name for the XML elements is taken from, in order of preference:
//     - the tag on the XMLName field, if the data is a struct
//     - the value of the XMLName field of type xml.Name
//     - the tag of the struct field used to obtain the data
//     - the name of the struct field used to obtain the data
//     - the name of the marshalled type
//
// The XML element for a struct contains marshalled elements for each of the
// exported fields of the struct, with these exceptions:
//     - the XMLName field, described above, is omitted.
//     - a field with tag "-" is omitted.
//     - a field with tag "name,attr" becomes an attribute with
//       the given name in the XML element.
//     - a field with tag ",attr" becomes an attribute with the
//       field name in the in the XML element.
//     - a field with tag ",chardata" is written as character data,
//       not as an XML element.
//     - a field with tag ",innerxml" is written verbatim, not subject
//       to the usual marshalling procedure.
//     - a field with tag ",comment" is written as an XML comment, not
//       subject to the usual marshalling procedure. It must not contain
//       the "--" string within it.
//
// If a field uses a tag "a>b>c", then the element c will be nested inside
// parent elements a and b.  Fields that appear next to each other that name
// the same parent will be enclosed in one XML element.  For example:
//
//	type Result struct {
//		XMLName   xml.Name `xml:"result"`
//		Id        int      `xml:"id,attr"`
//		FirstName string   `xml:"person>name>first"`
//		LastName  string   `xml:"person>name>last"`
//		Age       int      `xml:"person>age"`
//	}
//
//	xml.Marshal(&Result{Id: 13, FirstName: "John", LastName: "Doe", Age: 42})
//
// would be marshalled as:
//
//	<result>
//		<person id="13">
//			<name>
//				<first>John</first>
//				<last>Doe</last>
//			</name>
//			<age>42</age>
//		</person>
//	</result>
//
// Marshal will return an error if asked to marshal a channel, function, or map.
func Marshal(v interface{}) ([]byte, error) {
	var b bytes.Buffer
	if err := NewEncoder(&b).Encode(v); err != nil {
		return nil, err
	}
	return b.Bytes(), nil
}

// An Encoder writes XML data to an output stream.
type Encoder struct {
	printer
}

// NewEncoder returns a new encoder that writes to w.
func NewEncoder(w io.Writer) *Encoder {
	return &Encoder{printer{bufio.NewWriter(w)}}
}

// Encode writes the XML encoding of v to the stream.
//
// See the documentation for Marshal for details about the conversion
// of Go values to XML.
func (enc *Encoder) Encode(v interface{}) error {
	err := enc.marshalValue(reflect.ValueOf(v), nil)
	enc.Flush()
	return err
}

type printer struct {
	*bufio.Writer
}

func (p *printer) marshalValue(val reflect.Value, finfo *fieldInfo) error {
	if !val.IsValid() {
		return nil
	}

	kind := val.Kind()
	typ := val.Type()

	// Drill into pointers/interfaces
	if kind == reflect.Ptr || kind == reflect.Interface {
		if val.IsNil() {
			return nil
		}
		return p.marshalValue(val.Elem(), finfo)
	}

	// Slices and arrays iterate over the elements. They do not have an enclosing tag.
	if (kind == reflect.Slice || kind == reflect.Array) && typ.Elem().Kind() != reflect.Uint8 {
		for i, n := 0, val.Len(); i < n; i++ {
			if err := p.marshalValue(val.Index(i), finfo); err != nil {
				return err
			}
		}
		return nil
	}

	tinfo, err := getTypeInfo(typ)
	if err != nil {
		return err
	}

	// Precedence for the XML element name is:
	// 1. XMLName field in underlying struct;
	// 2. field name/tag in the struct field; and
	// 3. type name
	var xmlns, name string
	if tinfo.xmlname != nil {
		xmlname := tinfo.xmlname
		if xmlname.name != "" {
			xmlns, name = xmlname.xmlns, xmlname.name
		} else if v, ok := val.FieldByIndex(xmlname.idx).Interface().(Name); ok && v.Local != "" {
			xmlns, name = v.Space, v.Local
		}
	}
	if name == "" && finfo != nil {
		xmlns, name = finfo.xmlns, finfo.name
	}
	if name == "" {
		name = typ.Name()
		if name == "" {
			return &UnsupportedTypeError{typ}
		}
	}

	p.WriteByte('<')
	p.WriteString(name)

	if xmlns != "" {
		p.WriteString(` xmlns="`)
		// TODO: EscapeString, to avoid the allocation.
		Escape(p, []byte(xmlns))
		p.WriteByte('"')
	}

	// Attributes
	for i := range tinfo.fields {
		finfo := &tinfo.fields[i]
		if finfo.flags&fAttr == 0 {
			continue
		}
		fv := val.FieldByIndex(finfo.idx)
		switch fv.Kind() {
		case reflect.String, reflect.Array, reflect.Slice:
			// TODO: Should we really do this once ,omitempty is in?
			if fv.Len() == 0 {
				continue
			}
		}
		p.WriteByte(' ')
		p.WriteString(finfo.name)
		p.WriteString(`="`)
		if err := p.marshalSimple(fv.Type(), fv); err != nil {
			return err
		}
		p.WriteByte('"')
	}
	p.WriteByte('>')

	if val.Kind() == reflect.Struct {
		err = p.marshalStruct(tinfo, val)
	} else {
		err = p.marshalSimple(typ, val)
	}
	if err != nil {
		return err
	}

	p.WriteByte('<')
	p.WriteByte('/')
	p.WriteString(name)
	p.WriteByte('>')

	return nil
}

func (p *printer) marshalSimple(typ reflect.Type, val reflect.Value) error {
	switch val.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		p.WriteString(strconv.FormatInt(val.Int(), 10))
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		p.WriteString(strconv.FormatUint(val.Uint(), 10))
	case reflect.Float32, reflect.Float64:
		p.WriteString(strconv.FormatFloat(val.Float(), 'g', -1, 64))
	case reflect.String:
		// TODO: Add EscapeString.
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
		Escape(p, val.Bytes())
	default:
		return &UnsupportedTypeError{typ}
	}
	return nil
}

var ddBytes = []byte("--")

func (p *printer) marshalStruct(tinfo *typeInfo, val reflect.Value) error {
	s := parentStack{printer: p}
	for i := range tinfo.fields {
		finfo := &tinfo.fields[i]
		if finfo.flags&(fAttr|fAny) != 0 {
			continue
		}
		vf := val.FieldByIndex(finfo.idx)
		switch finfo.flags & fMode {
		case fCharData:
			switch vf.Kind() {
			case reflect.String:
				Escape(p, []byte(vf.String()))
			case reflect.Slice:
				if elem, ok := vf.Interface().([]byte); ok {
					Escape(p, elem)
				}
			}
			continue

		case fComment:
			k := vf.Kind()
			if !(k == reflect.String || k == reflect.Slice && vf.Type().Elem().Kind() == reflect.Uint8) {
				return fmt.Errorf("xml: bad type for comment field of %s", val.Type())
			}
			if vf.Len() == 0 {
				continue
			}
			p.WriteString("<!--")
			dashDash := false
			dashLast := false
			switch k {
			case reflect.String:
				s := vf.String()
				dashDash = strings.Index(s, "--") >= 0
				dashLast = s[len(s)-1] == '-'
				if !dashDash {
					p.WriteString(s)
				}
			case reflect.Slice:
				b := vf.Bytes()
				dashDash = bytes.Index(b, ddBytes) >= 0
				dashLast = b[len(b)-1] == '-'
				if !dashDash {
					p.Write(b)
				}
			default:
				panic("can't happen")
			}
			if dashDash {
				return fmt.Errorf(`xml: comments must not contain "--"`)
			}
			if dashLast {
				// "--->" is invalid grammar. Make it "- -->"
				p.WriteByte(' ')
			}
			p.WriteString("-->")
			continue

		case fInnerXml:
			iface := vf.Interface()
			switch raw := iface.(type) {
			case []byte:
				p.Write(raw)
				continue
			case string:
				p.WriteString(raw)
				continue
			}

		case fElement:
			s.trim(finfo.parents)
			if len(finfo.parents) > len(s.stack) {
				if vf.Kind() != reflect.Ptr && vf.Kind() != reflect.Interface || !vf.IsNil() {
					s.push(finfo.parents[len(s.stack):])
				}
			}
		}
		if err := p.marshalValue(vf, finfo); err != nil {
			return err
		}
	}
	s.trim(nil)
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

// A MarshalXMLError is returned when Marshal encounters a type
// that cannot be converted into XML.
type UnsupportedTypeError struct {
	Type reflect.Type
}

func (e *UnsupportedTypeError) Error() string {
	return "xml: unsupported type: " + e.Type.String()
}
