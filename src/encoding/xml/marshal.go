// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package xml

import (
	"bufio"
	"bytes"
	"encoding"
	"fmt"
	"io"
	"reflect"
	"strconv"
	"strings"
)

const (
	// Header is a generic XML header suitable for use with the output of Marshal.
	// This is not automatically added to any output of this package,
	// it is provided as a convenience.
	Header = `<?xml version="1.0" encoding="UTF-8"?>` + "\n"
)

// Marshal returns the XML encoding of v.
//
// Marshal handles an array or slice by marshaling each of the elements.
// Marshal handles a pointer by marshaling the value it points at or, if the
// pointer is nil, by writing nothing. Marshal handles an interface value by
// marshaling the value it contains or, if the interface value is nil, by
// writing nothing. Marshal handles all other data by writing one or more XML
// elements containing the data.
//
// The name for the XML elements is taken from, in order of preference:
//     - the tag on the XMLName field, if the data is a struct
//     - the value of the XMLName field of type Name
//     - the tag of the struct field used to obtain the data
//     - the name of the struct field used to obtain the data
//     - the name of the marshaled type
//
// The XML element for a struct contains marshaled elements for each of the
// exported fields of the struct, with these exceptions:
//     - the XMLName field, described above, is omitted.
//     - a field with tag "-" is omitted.
//     - a field with tag "name,attr" becomes an attribute with
//       the given name in the XML element.
//     - a field with tag ",attr" becomes an attribute with the
//       field name in the XML element.
//     - a field with tag ",chardata" is written as character data,
//       not as an XML element.
//     - a field with tag ",cdata" is written as character data
//       wrapped in one or more <![CDATA[ ... ]]> tags, not as an XML element.
//     - a field with tag ",innerxml" is written verbatim, not subject
//       to the usual marshaling procedure.
//     - a field with tag ",comment" is written as an XML comment, not
//       subject to the usual marshaling procedure. It must not contain
//       the "--" string within it.
//     - a field with a tag including the "omitempty" option is omitted
//       if the field value is empty. The empty values are false, 0, any
//       nil pointer or interface value, and any array, slice, map, or
//       string of length zero.
//     - an anonymous struct field is handled as if the fields of its
//       value were part of the outer struct.
//     - a field implementing Marshaler is written by calling its MarshalXML
//       method.
//     - a field implementing encoding.TextMarshaler is written by encoding the
//       result of its MarshalText method as text.
//
// If a field uses a tag "a>b>c", then the element c will be nested inside
// parent elements a and b. Fields that appear next to each other that name
// the same parent will be enclosed in one XML element.
//
// If the XML name for a struct field is defined by both the field tag and the
// struct's XMLName field, the names must match.
//
// See MarshalIndent for an example.
//
// Marshal will return an error if asked to marshal a channel, function, or map.
func Marshal(v interface{}) ([]byte, error) {
	var b bytes.Buffer
	if err := NewEncoder(&b).Encode(v); err != nil {
		return nil, err
	}
	return b.Bytes(), nil
}

// Marshaler is the interface implemented by objects that can marshal
// themselves into valid XML elements.
//
// MarshalXML encodes the receiver as zero or more XML elements.
// By convention, arrays or slices are typically encoded as a sequence
// of elements, one per entry.
// Using start as the element tag is not required, but doing so
// will enable Unmarshal to match the XML elements to the correct
// struct field.
// One common implementation strategy is to construct a separate
// value with a layout corresponding to the desired XML and then
// to encode it using e.EncodeElement.
// Another common strategy is to use repeated calls to e.EncodeToken
// to generate the XML output one token at a time.
// The sequence of encoded tokens must make up zero or more valid
// XML elements.
type Marshaler interface {
	MarshalXML(e *Encoder, start StartElement) error
}

// MarshalerAttr is the interface implemented by objects that can marshal
// themselves into valid XML attributes.
//
// MarshalXMLAttr returns an XML attribute with the encoded value of the receiver.
// Using name as the attribute name is not required, but doing so
// will enable Unmarshal to match the attribute to the correct
// struct field.
// If MarshalXMLAttr returns the zero attribute Attr{}, no attribute
// will be generated in the output.
// MarshalXMLAttr is used only for struct fields with the
// "attr" option in the field tag.
type MarshalerAttr interface {
	MarshalXMLAttr(name Name) (Attr, error)
}

// MarshalIndent works like Marshal, but each XML element begins on a new
// indented line that starts with prefix and is followed by one or more
// copies of indent according to the nesting depth.
func MarshalIndent(v interface{}, prefix, indent string) ([]byte, error) {
	var b bytes.Buffer
	enc := NewEncoder(&b)
	enc.Indent(prefix, indent)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return b.Bytes(), nil
}

// An Encoder writes XML data to an output stream.
type Encoder struct {
	p printer
}

// NewEncoder returns a new encoder that writes to w.
func NewEncoder(w io.Writer) *Encoder {
	e := &Encoder{printer{Writer: bufio.NewWriter(w)}}
	e.p.encoder = e
	return e
}

// Indent sets the encoder to generate XML in which each element
// begins on a new indented line that starts with prefix and is followed by
// one or more copies of indent according to the nesting depth.
func (enc *Encoder) Indent(prefix, indent string) {
	enc.p.prefix = prefix
	enc.p.indent = indent
}

// Encode writes the XML encoding of v to the stream.
//
// See the documentation for Marshal for details about the conversion
// of Go values to XML.
//
// Encode calls Flush before returning.
func (enc *Encoder) Encode(v interface{}) error {
	err := enc.p.marshalValue(reflect.ValueOf(v), nil, nil)
	if err != nil {
		return err
	}
	return enc.p.Flush()
}

// EncodeElement writes the XML encoding of v to the stream,
// using start as the outermost tag in the encoding.
//
// See the documentation for Marshal for details about the conversion
// of Go values to XML.
//
// EncodeElement calls Flush before returning.
func (enc *Encoder) EncodeElement(v interface{}, start StartElement) error {
	err := enc.p.marshalValue(reflect.ValueOf(v), nil, &start)
	if err != nil {
		return err
	}
	return enc.p.Flush()
}

var (
	begComment  = []byte("<!--")
	endComment  = []byte("-->")
	endProcInst = []byte("?>")
)

// EncodeToken writes the given XML token to the stream.
// It returns an error if StartElement and EndElement tokens are not properly matched.
//
// EncodeToken does not call Flush, because usually it is part of a larger operation
// such as Encode or EncodeElement (or a custom Marshaler's MarshalXML invoked
// during those), and those will call Flush when finished.
// Callers that create an Encoder and then invoke EncodeToken directly, without
// using Encode or EncodeElement, need to call Flush when finished to ensure
// that the XML is written to the underlying writer.
//
// EncodeToken allows writing a ProcInst with Target set to "xml" only as the first token
// in the stream.
func (enc *Encoder) EncodeToken(t Token) error {

	p := &enc.p
	switch t := t.(type) {
	case StartElement:
		if err := p.writeStart(&t); err != nil {
			return err
		}
	case EndElement:
		if err := p.writeEnd(t.Name); err != nil {
			return err
		}
	case CharData:
		escapeText(p, t, false)
	case Comment:
		if bytes.Contains(t, endComment) {
			return fmt.Errorf("xml: EncodeToken of Comment containing --> marker")
		}
		p.WriteString("<!--")
		p.Write(t)
		p.WriteString("-->")
		return p.cachedWriteError()
	case ProcInst:
		// First token to be encoded which is also a ProcInst with target of xml
		// is the xml declaration. The only ProcInst where target of xml is allowed.
		if t.Target == "xml" && p.Buffered() != 0 {
			return fmt.Errorf("xml: EncodeToken of ProcInst xml target only valid for xml declaration, first token encoded")
		}
		if !isNameString(t.Target) {
			return fmt.Errorf("xml: EncodeToken of ProcInst with invalid Target")
		}
		if bytes.Contains(t.Inst, endProcInst) {
			return fmt.Errorf("xml: EncodeToken of ProcInst containing ?> marker")
		}
		p.WriteString("<?")
		p.WriteString(t.Target)
		if len(t.Inst) > 0 {
			p.WriteByte(' ')
			p.Write(t.Inst)
		}
		p.WriteString("?>")
	case Directive:
		if !isValidDirective(t) {
			return fmt.Errorf("xml: EncodeToken of Directive containing wrong < or > markers")
		}
		p.WriteString("<!")
		p.Write(t)
		p.WriteString(">")
	default:
		return fmt.Errorf("xml: EncodeToken of invalid token type")

	}
	return p.cachedWriteError()
}

// isValidDirective reports whether dir is a valid directive text,
// meaning angle brackets are matched, ignoring comments and strings.
func isValidDirective(dir Directive) bool {
	var (
		depth     int
		inquote   uint8
		incomment bool
	)
	for i, c := range dir {
		switch {
		case incomment:
			if c == '>' {
				if n := 1 + i - len(endComment); n >= 0 && bytes.Equal(dir[n:i+1], endComment) {
					incomment = false
				}
			}
			// Just ignore anything in comment
		case inquote != 0:
			if c == inquote {
				inquote = 0
			}
			// Just ignore anything within quotes
		case c == '\'' || c == '"':
			inquote = c
		case c == '<':
			if i+len(begComment) < len(dir) && bytes.Equal(dir[i:i+len(begComment)], begComment) {
				incomment = true
			} else {
				depth++
			}
		case c == '>':
			if depth == 0 {
				return false
			}
			depth--
		}
	}
	return depth == 0 && inquote == 0 && !incomment
}

// Flush flushes any buffered XML to the underlying writer.
// See the EncodeToken documentation for details about when it is necessary.
func (enc *Encoder) Flush() error {
	return enc.p.Flush()
}

type printer struct {
	*bufio.Writer
	encoder    *Encoder
	seq        int
	indent     string
	prefix     string
	depth      int
	indentedIn bool
	putNewline bool
	attrNS     map[string]string // map prefix -> name space
	attrPrefix map[string]string // map name space -> prefix
	prefixes   []string
	tags       []Name
}

// createAttrPrefix finds the name space prefix attribute to use for the given name space,
// defining a new prefix if necessary. It returns the prefix.
func (p *printer) createAttrPrefix(url string) string {
	if prefix := p.attrPrefix[url]; prefix != "" {
		return prefix
	}

	// The "http://www.w3.org/XML/1998/namespace" name space is predefined as "xml"
	// and must be referred to that way.
	// (The "http://www.w3.org/2000/xmlns/" name space is also predefined as "xmlns",
	// but users should not be trying to use that one directly - that's our job.)
	if url == xmlURL {
		return xmlPrefix
	}

	// Need to define a new name space.
	if p.attrPrefix == nil {
		p.attrPrefix = make(map[string]string)
		p.attrNS = make(map[string]string)
	}

	// Pick a name. We try to use the final element of the path
	// but fall back to _.
	prefix := strings.TrimRight(url, "/")
	if i := strings.LastIndex(prefix, "/"); i >= 0 {
		prefix = prefix[i+1:]
	}
	if prefix == "" || !isName([]byte(prefix)) || strings.Contains(prefix, ":") {
		prefix = "_"
	}
	if strings.HasPrefix(prefix, "xml") {
		// xmlanything is reserved.
		prefix = "_" + prefix
	}
	if p.attrNS[prefix] != "" {
		// Name is taken. Find a better one.
		for p.seq++; ; p.seq++ {
			if id := prefix + "_" + strconv.Itoa(p.seq); p.attrNS[id] == "" {
				prefix = id
				break
			}
		}
	}

	p.attrPrefix[url] = prefix
	p.attrNS[prefix] = url

	p.WriteString(`xmlns:`)
	p.WriteString(prefix)
	p.WriteString(`="`)
	EscapeText(p, []byte(url))
	p.WriteString(`" `)

	p.prefixes = append(p.prefixes, prefix)

	return prefix
}

// deleteAttrPrefix removes an attribute name space prefix.
func (p *printer) deleteAttrPrefix(prefix string) {
	delete(p.attrPrefix, p.attrNS[prefix])
	delete(p.attrNS, prefix)
}

func (p *printer) markPrefix() {
	p.prefixes = append(p.prefixes, "")
}

func (p *printer) popPrefix() {
	for len(p.prefixes) > 0 {
		prefix := p.prefixes[len(p.prefixes)-1]
		p.prefixes = p.prefixes[:len(p.prefixes)-1]
		if prefix == "" {
			break
		}
		p.deleteAttrPrefix(prefix)
	}
}

var (
	marshalerType     = reflect.TypeOf((*Marshaler)(nil)).Elem()
	marshalerAttrType = reflect.TypeOf((*MarshalerAttr)(nil)).Elem()
	textMarshalerType = reflect.TypeOf((*encoding.TextMarshaler)(nil)).Elem()
)

// marshalValue writes one or more XML elements representing val.
// If val was obtained from a struct field, finfo must have its details.
func (p *printer) marshalValue(val reflect.Value, finfo *fieldInfo, startTemplate *StartElement) error {
	if startTemplate != nil && startTemplate.Name.Local == "" {
		return fmt.Errorf("xml: EncodeElement of StartElement with missing name")
	}

	if !val.IsValid() {
		return nil
	}
	if finfo != nil && finfo.flags&fOmitEmpty != 0 && isEmptyValue(val) {
		return nil
	}

	// Drill into interfaces and pointers.
	// This can turn into an infinite loop given a cyclic chain,
	// but it matches the Go 1 behavior.
	for val.Kind() == reflect.Interface || val.Kind() == reflect.Ptr {
		if val.IsNil() {
			return nil
		}
		val = val.Elem()
	}

	kind := val.Kind()
	typ := val.Type()

	// Check for marshaler.
	if val.CanInterface() && typ.Implements(marshalerType) {
		return p.marshalInterface(val.Interface().(Marshaler), defaultStart(typ, finfo, startTemplate))
	}
	if val.CanAddr() {
		pv := val.Addr()
		if pv.CanInterface() && pv.Type().Implements(marshalerType) {
			return p.marshalInterface(pv.Interface().(Marshaler), defaultStart(pv.Type(), finfo, startTemplate))
		}
	}

	// Check for text marshaler.
	if val.CanInterface() && typ.Implements(textMarshalerType) {
		return p.marshalTextInterface(val.Interface().(encoding.TextMarshaler), defaultStart(typ, finfo, startTemplate))
	}
	if val.CanAddr() {
		pv := val.Addr()
		if pv.CanInterface() && pv.Type().Implements(textMarshalerType) {
			return p.marshalTextInterface(pv.Interface().(encoding.TextMarshaler), defaultStart(pv.Type(), finfo, startTemplate))
		}
	}

	// Slices and arrays iterate over the elements. They do not have an enclosing tag.
	if (kind == reflect.Slice || kind == reflect.Array) && typ.Elem().Kind() != reflect.Uint8 {
		for i, n := 0, val.Len(); i < n; i++ {
			if err := p.marshalValue(val.Index(i), finfo, startTemplate); err != nil {
				return err
			}
		}
		return nil
	}

	tinfo, err := getTypeInfo(typ)
	if err != nil {
		return err
	}

	// Create start element.
	// Precedence for the XML element name is:
	// 0. startTemplate
	// 1. XMLName field in underlying struct;
	// 2. field name/tag in the struct field; and
	// 3. type name
	var start StartElement

	if startTemplate != nil {
		start.Name = startTemplate.Name
		start.Attr = append(start.Attr, startTemplate.Attr...)
	} else if tinfo.xmlname != nil {
		xmlname := tinfo.xmlname
		if xmlname.name != "" {
			start.Name.Space, start.Name.Local = xmlname.xmlns, xmlname.name
		} else if v, ok := xmlname.value(val).Interface().(Name); ok && v.Local != "" {
			start.Name = v
		}
	}
	if start.Name.Local == "" && finfo != nil {
		start.Name.Space, start.Name.Local = finfo.xmlns, finfo.name
	}
	if start.Name.Local == "" {
		name := typ.Name()
		if name == "" {
			return &UnsupportedTypeError{typ}
		}
		start.Name.Local = name
	}

	// Attributes
	for i := range tinfo.fields {
		finfo := &tinfo.fields[i]
		if finfo.flags&fAttr == 0 {
			continue
		}
		fv := finfo.value(val)

		if finfo.flags&fOmitEmpty != 0 && isEmptyValue(fv) {
			continue
		}

		if fv.Kind() == reflect.Interface && fv.IsNil() {
			continue
		}

		name := Name{Space: finfo.xmlns, Local: finfo.name}
		if err := p.marshalAttr(&start, name, fv); err != nil {
			return err
		}
	}

	if err := p.writeStart(&start); err != nil {
		return err
	}

	if val.Kind() == reflect.Struct {
		err = p.marshalStruct(tinfo, val)
	} else {
		s, b, err1 := p.marshalSimple(typ, val)
		if err1 != nil {
			err = err1
		} else if b != nil {
			EscapeText(p, b)
		} else {
			p.EscapeString(s)
		}
	}
	if err != nil {
		return err
	}

	if err := p.writeEnd(start.Name); err != nil {
		return err
	}

	return p.cachedWriteError()
}

// marshalAttr marshals an attribute with the given name and value, adding to start.Attr.
func (p *printer) marshalAttr(start *StartElement, name Name, val reflect.Value) error {
	if val.CanInterface() && val.Type().Implements(marshalerAttrType) {
		attr, err := val.Interface().(MarshalerAttr).MarshalXMLAttr(name)
		if err != nil {
			return err
		}
		if attr.Name.Local != "" {
			start.Attr = append(start.Attr, attr)
		}
		return nil
	}

	if val.CanAddr() {
		pv := val.Addr()
		if pv.CanInterface() && pv.Type().Implements(marshalerAttrType) {
			attr, err := pv.Interface().(MarshalerAttr).MarshalXMLAttr(name)
			if err != nil {
				return err
			}
			if attr.Name.Local != "" {
				start.Attr = append(start.Attr, attr)
			}
			return nil
		}
	}

	if val.CanInterface() && val.Type().Implements(textMarshalerType) {
		text, err := val.Interface().(encoding.TextMarshaler).MarshalText()
		if err != nil {
			return err
		}
		start.Attr = append(start.Attr, Attr{name, string(text)})
		return nil
	}

	if val.CanAddr() {
		pv := val.Addr()
		if pv.CanInterface() && pv.Type().Implements(textMarshalerType) {
			text, err := pv.Interface().(encoding.TextMarshaler).MarshalText()
			if err != nil {
				return err
			}
			start.Attr = append(start.Attr, Attr{name, string(text)})
			return nil
		}
	}

	// Dereference or skip nil pointer, interface values.
	switch val.Kind() {
	case reflect.Ptr, reflect.Interface:
		if val.IsNil() {
			return nil
		}
		val = val.Elem()
	}

	// Walk slices.
	if val.Kind() == reflect.Slice && val.Type().Elem().Kind() != reflect.Uint8 {
		n := val.Len()
		for i := 0; i < n; i++ {
			if err := p.marshalAttr(start, name, val.Index(i)); err != nil {
				return err
			}
		}
		return nil
	}

	if val.Type() == attrType {
		start.Attr = append(start.Attr, val.Interface().(Attr))
		return nil
	}

	s, b, err := p.marshalSimple(val.Type(), val)
	if err != nil {
		return err
	}
	if b != nil {
		s = string(b)
	}
	start.Attr = append(start.Attr, Attr{name, s})
	return nil
}

// defaultStart returns the default start element to use,
// given the reflect type, field info, and start template.
func defaultStart(typ reflect.Type, finfo *fieldInfo, startTemplate *StartElement) StartElement {
	var start StartElement
	// Precedence for the XML element name is as above,
	// except that we do not look inside structs for the first field.
	if startTemplate != nil {
		start.Name = startTemplate.Name
		start.Attr = append(start.Attr, startTemplate.Attr...)
	} else if finfo != nil && finfo.name != "" {
		start.Name.Local = finfo.name
		start.Name.Space = finfo.xmlns
	} else if typ.Name() != "" {
		start.Name.Local = typ.Name()
	} else {
		// Must be a pointer to a named type,
		// since it has the Marshaler methods.
		start.Name.Local = typ.Elem().Name()
	}
	return start
}

// marshalInterface marshals a Marshaler interface value.
func (p *printer) marshalInterface(val Marshaler, start StartElement) error {
	// Push a marker onto the tag stack so that MarshalXML
	// cannot close the XML tags that it did not open.
	p.tags = append(p.tags, Name{})
	n := len(p.tags)

	err := val.MarshalXML(p.encoder, start)
	if err != nil {
		return err
	}

	// Make sure MarshalXML closed all its tags. p.tags[n-1] is the mark.
	if len(p.tags) > n {
		return fmt.Errorf("xml: %s.MarshalXML wrote invalid XML: <%s> not closed", receiverType(val), p.tags[len(p.tags)-1].Local)
	}
	p.tags = p.tags[:n-1]
	return nil
}

// marshalTextInterface marshals a TextMarshaler interface value.
func (p *printer) marshalTextInterface(val encoding.TextMarshaler, start StartElement) error {
	if err := p.writeStart(&start); err != nil {
		return err
	}
	text, err := val.MarshalText()
	if err != nil {
		return err
	}
	EscapeText(p, text)
	return p.writeEnd(start.Name)
}

// writeStart writes the given start element.
func (p *printer) writeStart(start *StartElement) error {
	if start.Name.Local == "" {
		return fmt.Errorf("xml: start tag with no name")
	}

	p.tags = append(p.tags, start.Name)
	p.markPrefix()

	p.writeIndent(1)
	p.WriteByte('<')
	p.WriteString(start.Name.Local)

	if start.Name.Space != "" {
		p.WriteString(` xmlns="`)
		p.EscapeString(start.Name.Space)
		p.WriteByte('"')
	}

	// Attributes
	for _, attr := range start.Attr {
		name := attr.Name
		if name.Local == "" {
			continue
		}
		p.WriteByte(' ')
		if name.Space != "" {
			p.WriteString(p.createAttrPrefix(name.Space))
			p.WriteByte(':')
		}
		p.WriteString(name.Local)
		p.WriteString(`="`)
		p.EscapeString(attr.Value)
		p.WriteByte('"')
	}
	p.WriteByte('>')
	return nil
}

func (p *printer) writeEnd(name Name) error {
	if name.Local == "" {
		return fmt.Errorf("xml: end tag with no name")
	}
	if len(p.tags) == 0 || p.tags[len(p.tags)-1].Local == "" {
		return fmt.Errorf("xml: end tag </%s> without start tag", name.Local)
	}
	if top := p.tags[len(p.tags)-1]; top != name {
		if top.Local != name.Local {
			return fmt.Errorf("xml: end tag </%s> does not match start tag <%s>", name.Local, top.Local)
		}
		return fmt.Errorf("xml: end tag </%s> in namespace %s does not match start tag <%s> in namespace %s", name.Local, name.Space, top.Local, top.Space)
	}
	p.tags = p.tags[:len(p.tags)-1]

	p.writeIndent(-1)
	p.WriteByte('<')
	p.WriteByte('/')
	p.WriteString(name.Local)
	p.WriteByte('>')
	p.popPrefix()
	return nil
}

func (p *printer) marshalSimple(typ reflect.Type, val reflect.Value) (string, []byte, error) {
	switch val.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return strconv.FormatInt(val.Int(), 10), nil, nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return strconv.FormatUint(val.Uint(), 10), nil, nil
	case reflect.Float32, reflect.Float64:
		return strconv.FormatFloat(val.Float(), 'g', -1, val.Type().Bits()), nil, nil
	case reflect.String:
		return val.String(), nil, nil
	case reflect.Bool:
		return strconv.FormatBool(val.Bool()), nil, nil
	case reflect.Array:
		if typ.Elem().Kind() != reflect.Uint8 {
			break
		}
		// [...]byte
		var bytes []byte
		if val.CanAddr() {
			bytes = val.Slice(0, val.Len()).Bytes()
		} else {
			bytes = make([]byte, val.Len())
			reflect.Copy(reflect.ValueOf(bytes), val)
		}
		return "", bytes, nil
	case reflect.Slice:
		if typ.Elem().Kind() != reflect.Uint8 {
			break
		}
		// []byte
		return "", val.Bytes(), nil
	}
	return "", nil, &UnsupportedTypeError{typ}
}

var ddBytes = []byte("--")

// indirect drills into interfaces and pointers, returning the pointed-at value.
// If it encounters a nil interface or pointer, indirect returns that nil value.
// This can turn into an infinite loop given a cyclic chain,
// but it matches the Go 1 behavior.
func indirect(vf reflect.Value) reflect.Value {
	for vf.Kind() == reflect.Interface || vf.Kind() == reflect.Ptr {
		if vf.IsNil() {
			return vf
		}
		vf = vf.Elem()
	}
	return vf
}

func (p *printer) marshalStruct(tinfo *typeInfo, val reflect.Value) error {
	s := parentStack{p: p}
	for i := range tinfo.fields {
		finfo := &tinfo.fields[i]
		if finfo.flags&fAttr != 0 {
			continue
		}
		vf := finfo.value(val)

		switch finfo.flags & fMode {
		case fCDATA, fCharData:
			emit := EscapeText
			if finfo.flags&fMode == fCDATA {
				emit = emitCDATA
			}
			if err := s.trim(finfo.parents); err != nil {
				return err
			}
			if vf.CanInterface() && vf.Type().Implements(textMarshalerType) {
				data, err := vf.Interface().(encoding.TextMarshaler).MarshalText()
				if err != nil {
					return err
				}
				if err := emit(p, data); err != nil {
					return err
				}
				continue
			}
			if vf.CanAddr() {
				pv := vf.Addr()
				if pv.CanInterface() && pv.Type().Implements(textMarshalerType) {
					data, err := pv.Interface().(encoding.TextMarshaler).MarshalText()
					if err != nil {
						return err
					}
					if err := emit(p, data); err != nil {
						return err
					}
					continue
				}
			}

			var scratch [64]byte
			vf = indirect(vf)
			switch vf.Kind() {
			case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
				if err := emit(p, strconv.AppendInt(scratch[:0], vf.Int(), 10)); err != nil {
					return err
				}
			case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
				if err := emit(p, strconv.AppendUint(scratch[:0], vf.Uint(), 10)); err != nil {
					return err
				}
			case reflect.Float32, reflect.Float64:
				if err := emit(p, strconv.AppendFloat(scratch[:0], vf.Float(), 'g', -1, vf.Type().Bits())); err != nil {
					return err
				}
			case reflect.Bool:
				if err := emit(p, strconv.AppendBool(scratch[:0], vf.Bool())); err != nil {
					return err
				}
			case reflect.String:
				if err := emit(p, []byte(vf.String())); err != nil {
					return err
				}
			case reflect.Slice:
				if elem, ok := vf.Interface().([]byte); ok {
					if err := emit(p, elem); err != nil {
						return err
					}
				}
			}
			continue

		case fComment:
			if err := s.trim(finfo.parents); err != nil {
				return err
			}
			vf = indirect(vf)
			k := vf.Kind()
			if !(k == reflect.String || k == reflect.Slice && vf.Type().Elem().Kind() == reflect.Uint8) {
				return fmt.Errorf("xml: bad type for comment field of %s", val.Type())
			}
			if vf.Len() == 0 {
				continue
			}
			p.writeIndent(0)
			p.WriteString("<!--")
			dashDash := false
			dashLast := false
			switch k {
			case reflect.String:
				s := vf.String()
				dashDash = strings.Contains(s, "--")
				dashLast = s[len(s)-1] == '-'
				if !dashDash {
					p.WriteString(s)
				}
			case reflect.Slice:
				b := vf.Bytes()
				dashDash = bytes.Contains(b, ddBytes)
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

		case fInnerXML:
			vf = indirect(vf)
			iface := vf.Interface()
			switch raw := iface.(type) {
			case []byte:
				p.Write(raw)
				continue
			case string:
				p.WriteString(raw)
				continue
			}

		case fElement, fElement | fAny:
			if err := s.trim(finfo.parents); err != nil {
				return err
			}
			if len(finfo.parents) > len(s.stack) {
				if vf.Kind() != reflect.Ptr && vf.Kind() != reflect.Interface || !vf.IsNil() {
					if err := s.push(finfo.parents[len(s.stack):]); err != nil {
						return err
					}
				}
			}
		}
		if err := p.marshalValue(vf, finfo, nil); err != nil {
			return err
		}
	}
	s.trim(nil)
	return p.cachedWriteError()
}

// return the bufio Writer's cached write error
func (p *printer) cachedWriteError() error {
	_, err := p.Write(nil)
	return err
}

func (p *printer) writeIndent(depthDelta int) {
	if len(p.prefix) == 0 && len(p.indent) == 0 {
		return
	}
	if depthDelta < 0 {
		p.depth--
		if p.indentedIn {
			p.indentedIn = false
			return
		}
		p.indentedIn = false
	}
	if p.putNewline {
		p.WriteByte('\n')
	} else {
		p.putNewline = true
	}
	if len(p.prefix) > 0 {
		p.WriteString(p.prefix)
	}
	if len(p.indent) > 0 {
		for i := 0; i < p.depth; i++ {
			p.WriteString(p.indent)
		}
	}
	if depthDelta > 0 {
		p.depth++
		p.indentedIn = true
	}
}

type parentStack struct {
	p     *printer
	stack []string
}

// trim updates the XML context to match the longest common prefix of the stack
// and the given parents. A closing tag will be written for every parent
// popped. Passing a zero slice or nil will close all the elements.
func (s *parentStack) trim(parents []string) error {
	split := 0
	for ; split < len(parents) && split < len(s.stack); split++ {
		if parents[split] != s.stack[split] {
			break
		}
	}
	for i := len(s.stack) - 1; i >= split; i-- {
		if err := s.p.writeEnd(Name{Local: s.stack[i]}); err != nil {
			return err
		}
	}
	s.stack = s.stack[:split]
	return nil
}

// push adds parent elements to the stack and writes open tags.
func (s *parentStack) push(parents []string) error {
	for i := 0; i < len(parents); i++ {
		if err := s.p.writeStart(&StartElement{Name: Name{Local: parents[i]}}); err != nil {
			return err
		}
	}
	s.stack = append(s.stack, parents...)
	return nil
}

// UnsupportedTypeError is returned when Marshal encounters a type
// that cannot be converted into XML.
type UnsupportedTypeError struct {
	Type reflect.Type
}

func (e *UnsupportedTypeError) Error() string {
	return "xml: unsupported type: " + e.Type.String()
}

func isEmptyValue(v reflect.Value) bool {
	switch v.Kind() {
	case reflect.Array, reflect.Map, reflect.Slice, reflect.String:
		return v.Len() == 0
	case reflect.Bool:
		return !v.Bool()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return v.Int() == 0
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return v.Uint() == 0
	case reflect.Float32, reflect.Float64:
		return v.Float() == 0
	case reflect.Interface, reflect.Ptr:
		return v.IsNil()
	}
	return false
}
