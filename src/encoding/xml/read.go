// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package xml

import (
	"bytes"
	"encoding"
	"errors"
	"fmt"
	"reflect"
	"runtime"
	"strconv"
	"strings"
)

// BUG(rsc): Mapping between XML elements and data structures is inherently flawed:
// an XML element is an order-dependent collection of anonymous
// values, while a data structure is an order-independent collection
// of named values.
// See [encoding/json] for a textual representation more suitable
// to data structures.

// Unmarshal parses the XML-encoded data and stores the result in
// the value pointed to by v, which must be an arbitrary struct,
// slice, or string. Well-formed data that does not fit into v is
// discarded.
//
// Because Unmarshal uses the reflect package, it can only assign
// to exported (upper case) fields. Unmarshal uses a case-sensitive
// comparison to match XML element names to tag values and struct
// field names.
//
// Unmarshal maps an XML element to a struct using the following rules.
// In the rules, the tag of a field refers to the value associated with the
// key 'xml' in the struct field's tag (see the example above).
//
//   - If the struct has a field of type []byte or string with tag
//     ",innerxml", Unmarshal accumulates the raw XML nested inside the
//     element in that field. The rest of the rules still apply.
//
//   - If the struct has a field named XMLName of type Name,
//     Unmarshal records the element name in that field.
//
//   - If the XMLName field has an associated tag of the form
//     "name" or "namespace-URL name", the XML element must have
//     the given name (and, optionally, name space) or else Unmarshal
//     returns an error.
//
//   - If the XML element has an attribute whose name matches a
//     struct field name with an associated tag containing ",attr" or
//     the explicit name in a struct field tag of the form "name,attr",
//     Unmarshal records the attribute value in that field.
//
//   - If the XML element has an attribute not handled by the previous
//     rule and the struct has a field with an associated tag containing
//     ",any,attr", Unmarshal records the attribute value in the first
//     such field.
//
//   - If the XML element contains character data, that data is
//     accumulated in the first struct field that has tag ",chardata".
//     The struct field may have type []byte or string.
//     If there is no such field, the character data is discarded.
//
//   - If the XML element contains comments, they are accumulated in
//     the first struct field that has tag ",comment".  The struct
//     field may have type []byte or string. If there is no such
//     field, the comments are discarded.
//
//   - If the XML element contains a sub-element whose name matches
//     the prefix of a tag formatted as "a" or "a>b>c", unmarshal
//     will descend into the XML structure looking for elements with the
//     given names, and will map the innermost elements to that struct
//     field. A tag starting with ">" is equivalent to one starting
//     with the field name followed by ">".
//
//   - If the XML element contains a sub-element whose name matches
//     a struct field's XMLName tag and the struct field has no
//     explicit name tag as per the previous rule, unmarshal maps
//     the sub-element to that struct field.
//
//   - If the XML element contains a sub-element whose name matches a
//     field without any mode flags (",attr", ",chardata", etc), Unmarshal
//     maps the sub-element to that struct field.
//
//   - If the XML element contains a sub-element that hasn't matched any
//     of the above rules and the struct has a field with tag ",any",
//     unmarshal maps the sub-element to that struct field.
//
//   - An anonymous struct field is handled as if the fields of its
//     value were part of the outer struct.
//
//   - A struct field with tag "-" is never unmarshaled into.
//
// If Unmarshal encounters a field type that implements the Unmarshaler
// interface, Unmarshal calls its UnmarshalXML method to produce the value from
// the XML element.  Otherwise, if the value implements
// [encoding.TextUnmarshaler], Unmarshal calls that value's UnmarshalText method.
//
// Unmarshal maps an XML element to a string or []byte by saving the
// concatenation of that element's character data in the string or
// []byte. The saved []byte is never nil.
//
// Unmarshal maps an attribute value to a string or []byte by saving
// the value in the string or slice.
//
// Unmarshal maps an attribute value to an [Attr] by saving the attribute,
// including its name, in the Attr.
//
// Unmarshal maps an XML element or attribute value to a slice by
// extending the length of the slice and mapping the element or attribute
// to the newly created value.
//
// Unmarshal maps an XML element or attribute value to a bool by
// setting it to the boolean value represented by the string. Whitespace
// is trimmed and ignored.
//
// Unmarshal maps an XML element or attribute value to an integer or
// floating-point field by setting the field to the result of
// interpreting the string value in decimal. There is no check for
// overflow. Whitespace is trimmed and ignored.
//
// Unmarshal maps an XML element to a Name by recording the element
// name.
//
// Unmarshal maps an XML element to a pointer by setting the pointer
// to a freshly allocated value and then mapping the element to that value.
//
// A missing element or empty attribute value will be unmarshaled as a zero value.
// If the field is a slice, a zero value will be appended to the field. Otherwise, the
// field will be set to its zero value.
func Unmarshal(data []byte, v any) error {
	return NewDecoder(bytes.NewReader(data)).Decode(v)
}

// Decode works like [Unmarshal], except it reads the decoder
// stream to find the start element.
func (d *Decoder) Decode(v any) error {
	return d.DecodeElement(v, nil)
}

// DecodeElement works like [Unmarshal] except that it takes
// a pointer to the start XML element to decode into v.
// It is useful when a client reads some raw XML tokens itself
// but also wants to defer to [Unmarshal] for some elements.
func (d *Decoder) DecodeElement(v any, start *StartElement) error {
	val := reflect.ValueOf(v)
	if val.Kind() != reflect.Pointer {
		return errors.New("non-pointer passed to Unmarshal")
	}

	if val.IsNil() {
		return errors.New("nil pointer passed to Unmarshal")
	}
	return d.unmarshal(val.Elem(), start, 0)
}

// An UnmarshalError represents an error in the unmarshaling process.
type UnmarshalError string

func (e UnmarshalError) Error() string { return string(e) }

// Unmarshaler is the interface implemented by objects that can unmarshal
// an XML element description of themselves.
//
// UnmarshalXML decodes a single XML element
// beginning with the given start element.
// If it returns an error, the outer call to Unmarshal stops and
// returns that error.
// UnmarshalXML must consume exactly one XML element.
// One common implementation strategy is to unmarshal into
// a separate value with a layout matching the expected XML
// using d.DecodeElement, and then to copy the data from
// that value into the receiver.
// Another common strategy is to use d.Token to process the
// XML object one token at a time.
// UnmarshalXML may not use d.RawToken.
type Unmarshaler interface {
	UnmarshalXML(d *Decoder, start StartElement) error
}

// UnmarshalerAttr is the interface implemented by objects that can unmarshal
// an XML attribute description of themselves.
//
// UnmarshalXMLAttr decodes a single XML attribute.
// If it returns an error, the outer call to [Unmarshal] stops and
// returns that error.
// UnmarshalXMLAttr is used only for struct fields with the
// "attr" option in the field tag.
type UnmarshalerAttr interface {
	UnmarshalXMLAttr(attr Attr) error
}

// receiverType returns the receiver type to use in an expression like "%s.MethodName".
func receiverType(val any) string {
	t := reflect.TypeOf(val)
	if t.Name() != "" {
		return t.String()
	}
	return "(" + t.String() + ")"
}

// unmarshalInterface unmarshals a single XML element into val.
// start is the opening tag of the element.
func (d *Decoder) unmarshalInterface(val Unmarshaler, start *StartElement) error {
	// Record that decoder must stop at end tag corresponding to start.
	d.pushEOF()

	d.unmarshalDepth++
	err := val.UnmarshalXML(d, *start)
	d.unmarshalDepth--
	if err != nil {
		d.popEOF()
		return err
	}

	if !d.popEOF() {
		return fmt.Errorf("xml: %s.UnmarshalXML did not consume entire <%s> element", receiverType(val), start.Name.Local)
	}

	return nil
}

// unmarshalTextInterface unmarshals a single XML element into val.
// The chardata contained in the element (but not its children)
// is passed to the text unmarshaler.
func (d *Decoder) unmarshalTextInterface(val encoding.TextUnmarshaler) error {
	var buf []byte
	depth := 1
	for depth > 0 {
		t, err := d.Token()
		if err != nil {
			return err
		}
		switch t := t.(type) {
		case CharData:
			if depth == 1 {
				buf = append(buf, t...)
			}
		case StartElement:
			depth++
		case EndElement:
			depth--
		}
	}
	return val.UnmarshalText(buf)
}

// unmarshalAttr unmarshals a single XML attribute into val.
func (d *Decoder) unmarshalAttr(val reflect.Value, attr Attr) error {
	if val.Kind() == reflect.Pointer {
		if val.IsNil() {
			val.Set(reflect.New(val.Type().Elem()))
		}
		val = val.Elem()
	}
	if val.CanInterface() && val.Type().Implements(unmarshalerAttrType) {
		// This is an unmarshaler with a non-pointer receiver,
		// so it's likely to be incorrect, but we do what we're told.
		return val.Interface().(UnmarshalerAttr).UnmarshalXMLAttr(attr)
	}
	if val.CanAddr() {
		pv := val.Addr()
		if pv.CanInterface() && pv.Type().Implements(unmarshalerAttrType) {
			return pv.Interface().(UnmarshalerAttr).UnmarshalXMLAttr(attr)
		}
	}

	// Not an UnmarshalerAttr; try encoding.TextUnmarshaler.
	if val.CanInterface() && val.Type().Implements(textUnmarshalerType) {
		// This is an unmarshaler with a non-pointer receiver,
		// so it's likely to be incorrect, but we do what we're told.
		return val.Interface().(encoding.TextUnmarshaler).UnmarshalText([]byte(attr.Value))
	}
	if val.CanAddr() {
		pv := val.Addr()
		if pv.CanInterface() && pv.Type().Implements(textUnmarshalerType) {
			return pv.Interface().(encoding.TextUnmarshaler).UnmarshalText([]byte(attr.Value))
		}
	}

	if val.Kind() == reflect.Slice && val.Type().Elem().Kind() != reflect.Uint8 {
		// Slice of element values.
		// Grow slice.
		n := val.Len()
		val.Grow(1)
		val.SetLen(n + 1)

		// Recur to read element into slice.
		if err := d.unmarshalAttr(val.Index(n), attr); err != nil {
			val.SetLen(n)
			return err
		}
		return nil
	}

	if val.Type() == attrType {
		val.Set(reflect.ValueOf(attr))
		return nil
	}

	return copyValue(val, []byte(attr.Value))
}

var (
	attrType            = reflect.TypeFor[Attr]()
	unmarshalerType     = reflect.TypeFor[Unmarshaler]()
	unmarshalerAttrType = reflect.TypeFor[UnmarshalerAttr]()
	textUnmarshalerType = reflect.TypeFor[encoding.TextUnmarshaler]()
)

const (
	maxUnmarshalDepth     = 10000
	maxUnmarshalDepthWasm = 5000 // go.dev/issue/56498
)

var errUnmarshalDepth = errors.New("exceeded max depth")

// Unmarshal a single XML element into val.
func (d *Decoder) unmarshal(val reflect.Value, start *StartElement, depth int) error {
	if depth >= maxUnmarshalDepth || runtime.GOARCH == "wasm" && depth >= maxUnmarshalDepthWasm {
		return errUnmarshalDepth
	}
	// Find start element if we need it.
	if start == nil {
		for {
			tok, err := d.Token()
			if err != nil {
				return err
			}
			if t, ok := tok.(StartElement); ok {
				start = &t
				break
			}
		}
	}

	// Load value from interface, but only if the result will be
	// usefully addressable.
	if val.Kind() == reflect.Interface && !val.IsNil() {
		e := val.Elem()
		if e.Kind() == reflect.Pointer && !e.IsNil() {
			val = e
		}
	}

	if val.Kind() == reflect.Pointer {
		if val.IsNil() {
			val.Set(reflect.New(val.Type().Elem()))
		}
		val = val.Elem()
	}

	if val.CanInterface() && val.Type().Implements(unmarshalerType) {
		// This is an unmarshaler with a non-pointer receiver,
		// so it's likely to be incorrect, but we do what we're told.
		return d.unmarshalInterface(val.Interface().(Unmarshaler), start)
	}

	if val.CanAddr() {
		pv := val.Addr()
		if pv.CanInterface() && pv.Type().Implements(unmarshalerType) {
			return d.unmarshalInterface(pv.Interface().(Unmarshaler), start)
		}
	}

	if val.CanInterface() && val.Type().Implements(textUnmarshalerType) {
		return d.unmarshalTextInterface(val.Interface().(encoding.TextUnmarshaler))
	}

	if val.CanAddr() {
		pv := val.Addr()
		if pv.CanInterface() && pv.Type().Implements(textUnmarshalerType) {
			return d.unmarshalTextInterface(pv.Interface().(encoding.TextUnmarshaler))
		}
	}

	var (
		data         []byte
		saveData     reflect.Value
		comment      []byte
		saveComment  reflect.Value
		saveXML      reflect.Value
		saveXMLIndex int
		saveXMLData  []byte
		saveAny      reflect.Value
		sv           reflect.Value
		tinfo        *typeInfo
		err          error
	)

	switch v := val; v.Kind() {
	default:
		return errors.New("unknown type " + v.Type().String())

	case reflect.Interface:
		// TODO: For now, simply ignore the field. In the near
		//       future we may choose to unmarshal the start
		//       element on it, if not nil.
		return d.Skip()

	case reflect.Slice:
		typ := v.Type()
		if typ.Elem().Kind() == reflect.Uint8 {
			// []byte
			saveData = v
			break
		}

		// Slice of element values.
		// Grow slice.
		n := v.Len()
		v.Grow(1)
		v.SetLen(n + 1)

		// Recur to read element into slice.
		if err := d.unmarshal(v.Index(n), start, depth+1); err != nil {
			v.SetLen(n)
			return err
		}
		return nil

	case reflect.Bool, reflect.Float32, reflect.Float64, reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64, reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr, reflect.String:
		saveData = v

	case reflect.Struct:
		typ := v.Type()
		if typ == nameType {
			v.Set(reflect.ValueOf(start.Name))
			break
		}

		sv = v
		tinfo, err = getTypeInfo(typ)
		if err != nil {
			return err
		}

		// Validate and assign element name.
		if tinfo.xmlname != nil {
			finfo := tinfo.xmlname
			if finfo.name != "" && finfo.name != start.Name.Local {
				return UnmarshalError("expected element type <" + finfo.name + "> but have <" + start.Name.Local + ">")
			}
			if finfo.xmlns != "" && finfo.xmlns != start.Name.Space {
				e := "expected element <" + finfo.name + "> in name space " + finfo.xmlns + " but have "
				if start.Name.Space == "" {
					e += "no name space"
				} else {
					e += start.Name.Space
				}
				return UnmarshalError(e)
			}
			fv := finfo.value(sv, initNilPointers)
			if _, ok := fv.Interface().(Name); ok {
				fv.Set(reflect.ValueOf(start.Name))
			}
		}

		// Assign attributes.
		for _, a := range start.Attr {
			handled := false
			any := -1
			for i := range tinfo.fields {
				finfo := &tinfo.fields[i]
				switch finfo.flags & fMode {
				case fAttr:
					strv := finfo.value(sv, initNilPointers)
					if a.Name.Local == finfo.name && (finfo.xmlns == "" || finfo.xmlns == a.Name.Space) {
						if err := d.unmarshalAttr(strv, a); err != nil {
							return err
						}
						handled = true
					}

				case fAny | fAttr:
					if any == -1 {
						any = i
					}
				}
			}
			if !handled && any >= 0 {
				finfo := &tinfo.fields[any]
				strv := finfo.value(sv, initNilPointers)
				if err := d.unmarshalAttr(strv, a); err != nil {
					return err
				}
			}
		}

		// Determine whether we need to save character data or comments.
		for i := range tinfo.fields {
			finfo := &tinfo.fields[i]
			switch finfo.flags & fMode {
			case fCDATA, fCharData:
				if !saveData.IsValid() {
					saveData = finfo.value(sv, initNilPointers)
				}

			case fComment:
				if !saveComment.IsValid() {
					saveComment = finfo.value(sv, initNilPointers)
				}

			case fAny, fAny | fElement:
				if !saveAny.IsValid() {
					saveAny = finfo.value(sv, initNilPointers)
				}

			case fInnerXML:
				if !saveXML.IsValid() {
					saveXML = finfo.value(sv, initNilPointers)
					if d.saved == nil {
						saveXMLIndex = 0
						d.saved = new(bytes.Buffer)
					} else {
						saveXMLIndex = d.savedOffset()
					}
				}
			}
		}
	}

	// Find end element.
	// Process sub-elements along the way.
Loop:
	for {
		var savedOffset int
		if saveXML.IsValid() {
			savedOffset = d.savedOffset()
		}
		tok, err := d.Token()
		if err != nil {
			return err
		}
		switch t := tok.(type) {
		case StartElement:
			consumed := false
			if sv.IsValid() {
				// unmarshalPath can call unmarshal, so we need to pass the depth through so that
				// we can continue to enforce the maximum recursion limit.
				consumed, err = d.unmarshalPath(tinfo, sv, nil, &t, depth)
				if err != nil {
					return err
				}
				if !consumed && saveAny.IsValid() {
					consumed = true
					if err := d.unmarshal(saveAny, &t, depth+1); err != nil {
						return err
					}
				}
			}
			if !consumed {
				if err := d.Skip(); err != nil {
					return err
				}
			}

		case EndElement:
			if saveXML.IsValid() {
				saveXMLData = d.saved.Bytes()[saveXMLIndex:savedOffset]
				if saveXMLIndex == 0 {
					d.saved = nil
				}
			}
			break Loop

		case CharData:
			if saveData.IsValid() {
				data = append(data, t...)
			}

		case Comment:
			if saveComment.IsValid() {
				comment = append(comment, t...)
			}
		}
	}

	if saveData.IsValid() && saveData.CanInterface() && saveData.Type().Implements(textUnmarshalerType) {
		if err := saveData.Interface().(encoding.TextUnmarshaler).UnmarshalText(data); err != nil {
			return err
		}
		saveData = reflect.Value{}
	}

	if saveData.IsValid() && saveData.CanAddr() {
		pv := saveData.Addr()
		if pv.CanInterface() && pv.Type().Implements(textUnmarshalerType) {
			if err := pv.Interface().(encoding.TextUnmarshaler).UnmarshalText(data); err != nil {
				return err
			}
			saveData = reflect.Value{}
		}
	}

	if err := copyValue(saveData, data); err != nil {
		return err
	}

	switch t := saveComment; t.Kind() {
	case reflect.String:
		t.SetString(string(comment))
	case reflect.Slice:
		t.Set(reflect.ValueOf(comment))
	}

	switch t := saveXML; t.Kind() {
	case reflect.String:
		t.SetString(string(saveXMLData))
	case reflect.Slice:
		if t.Type().Elem().Kind() == reflect.Uint8 {
			t.Set(reflect.ValueOf(saveXMLData))
		}
	}

	return nil
}

func copyValue(dst reflect.Value, src []byte) (err error) {
	dst0 := dst

	if dst.Kind() == reflect.Pointer {
		if dst.IsNil() {
			dst.Set(reflect.New(dst.Type().Elem()))
		}
		dst = dst.Elem()
	}

	// Save accumulated data.
	switch dst.Kind() {
	case reflect.Invalid:
		// Probably a comment.
	default:
		return errors.New("cannot unmarshal into " + dst0.Type().String())
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		if len(src) == 0 {
			dst.SetInt(0)
			return nil
		}
		itmp, err := strconv.ParseInt(strings.TrimSpace(string(src)), 10, dst.Type().Bits())
		if err != nil {
			return err
		}
		dst.SetInt(itmp)
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		if len(src) == 0 {
			dst.SetUint(0)
			return nil
		}
		utmp, err := strconv.ParseUint(strings.TrimSpace(string(src)), 10, dst.Type().Bits())
		if err != nil {
			return err
		}
		dst.SetUint(utmp)
	case reflect.Float32, reflect.Float64:
		if len(src) == 0 {
			dst.SetFloat(0)
			return nil
		}
		ftmp, err := strconv.ParseFloat(strings.TrimSpace(string(src)), dst.Type().Bits())
		if err != nil {
			return err
		}
		dst.SetFloat(ftmp)
	case reflect.Bool:
		if len(src) == 0 {
			dst.SetBool(false)
			return nil
		}
		value, err := strconv.ParseBool(strings.TrimSpace(string(src)))
		if err != nil {
			return err
		}
		dst.SetBool(value)
	case reflect.String:
		dst.SetString(string(src))
	case reflect.Slice:
		if len(src) == 0 {
			// non-nil to flag presence
			src = []byte{}
		}
		dst.SetBytes(src)
	}
	return nil
}

// unmarshalPath walks down an XML structure looking for wanted
// paths, and calls unmarshal on them.
// The consumed result tells whether XML elements have been consumed
// from the Decoder until start's matching end element, or if it's
// still untouched because start is uninteresting for sv's fields.
func (d *Decoder) unmarshalPath(tinfo *typeInfo, sv reflect.Value, parents []string, start *StartElement, depth int) (consumed bool, err error) {
	recurse := false
Loop:
	for i := range tinfo.fields {
		finfo := &tinfo.fields[i]
		if finfo.flags&fElement == 0 || len(finfo.parents) < len(parents) || finfo.xmlns != "" && finfo.xmlns != start.Name.Space {
			continue
		}
		for j := range parents {
			if parents[j] != finfo.parents[j] {
				continue Loop
			}
		}
		if len(finfo.parents) == len(parents) && finfo.name == start.Name.Local {
			// It's a perfect match, unmarshal the field.
			return true, d.unmarshal(finfo.value(sv, initNilPointers), start, depth+1)
		}
		if len(finfo.parents) > len(parents) && finfo.parents[len(parents)] == start.Name.Local {
			// It's a prefix for the field. Break and recurse
			// since it's not ok for one field path to be itself
			// the prefix for another field path.
			recurse = true

			// We can reuse the same slice as long as we
			// don't try to append to it.
			parents = finfo.parents[:len(parents)+1]
			break
		}
	}
	if !recurse {
		// We have no business with this element.
		return false, nil
	}
	// The element is not a perfect match for any field, but one
	// or more fields have the path to this element as a parent
	// prefix. Recurse and attempt to match these.
	for {
		var tok Token
		tok, err = d.Token()
		if err != nil {
			return true, err
		}
		switch t := tok.(type) {
		case StartElement:
			// the recursion depth of unmarshalPath is limited to the path length specified
			// by the struct field tag, so we don't increment the depth here.
			consumed2, err := d.unmarshalPath(tinfo, sv, parents, &t, depth)
			if err != nil {
				return true, err
			}
			if !consumed2 {
				if err := d.Skip(); err != nil {
					return true, err
				}
			}
		case EndElement:
			return true, nil
		}
	}
}

// Skip reads tokens until it has consumed the end element
// matching the most recent start element already consumed,
// skipping nested structures.
// It returns nil if it finds an end element matching the start
// element; otherwise it returns an error describing the problem.
func (d *Decoder) Skip() error {
	var depth int64
	for {
		tok, err := d.Token()
		if err != nil {
			return err
		}
		switch tok.(type) {
		case StartElement:
			depth++
		case EndElement:
			if depth == 0 {
				return nil
			}
			depth--
		}
	}
}
