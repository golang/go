// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package xml

import (
	"bytes"
	"errors"
	"io"
	"reflect"
	"strconv"
	"strings"
)

// BUG(rsc): Mapping between XML elements and data structures is inherently flawed:
// an XML element is an order-dependent collection of anonymous
// values, while a data structure is an order-independent collection
// of named values.
// See package json for a textual representation more suitable
// to data structures.

// Unmarshal parses an XML element from r and uses the
// reflect library to fill in an arbitrary struct, slice, or string
// pointed at by val.  Well-formed data that does not fit
// into val is discarded.
//
// For example, given these definitions:
//
//	type Email struct {
//		Where string `xml:",attr"`
//		Addr  string
//	}
//
//	type Result struct {
//		XMLName xml.Name `xml:"result"`
//		Name	string
//		Phone	string
//		Email	[]Email
//		Groups  []string `xml:"group>value"`
//	}
//
//	result := Result{Name: "name", Phone: "phone", Email: nil}
//
// unmarshalling the XML input
//
//	<result>
//		<email where="home">
//			<addr>gre@example.com</addr>
//		</email>
//		<email where='work'>
//			<addr>gre@work.com</addr>
//		</email>
//		<name>Grace R. Emlin</name>
// 		<group>
// 			<value>Friends</value>
// 			<value>Squash</value>
// 		</group>
//		<address>123 Main Street</address>
//	</result>
//
// via Unmarshal(r, &result) is equivalent to assigning
//
//	r = Result{
//		xml.Name{Local: "result"},
//		"Grace R. Emlin", // name
//		"phone",	  // no phone given
//		[]Email{
//			Email{"home", "gre@example.com"},
//			Email{"work", "gre@work.com"},
//		},
//		[]string{"Friends", "Squash"},
//	}
//
// Note that the field r.Phone has not been modified and
// that the XML <address> element was discarded. Also, the field
// Groups was assigned considering the element path provided in the
// field tag.
//
// Because Unmarshal uses the reflect package, it can only assign
// to exported (upper case) fields.  Unmarshal uses a case-insensitive
// comparison to match XML element names to struct field names.
//
// Unmarshal maps an XML element to a struct using the following rules.
// In the rules, the tag of a field refers to the value associated with the
// key 'xml' in the struct field's tag (see the example above).
//
//   * If the struct has a field of type []byte or string with tag
//      ",innerxml", Unmarshal accumulates the raw XML nested inside the
//      element in that field.  The rest of the rules still apply.
//
//   * If the struct has a field named XMLName of type xml.Name,
//      Unmarshal records the element name in that field.
//
//   * If the XMLName field has an associated tag of the form
//      "name" or "namespace-URL name", the XML element must have
//      the given name (and, optionally, name space) or else Unmarshal
//      returns an error.
//
//   * If the XML element has an attribute whose name matches a
//      struct field name with an associated tag containing ",attr" or
//      the explicit name in a struct field tag of the form "name,attr",
//      Unmarshal records the attribute value in that field.
//
//   * If the XML element contains character data, that data is
//      accumulated in the first struct field that has tag "chardata".
//      The struct field may have type []byte or string.
//      If there is no such field, the character data is discarded.
//
//   * If the XML element contains comments, they are accumulated in
//      the first struct field that has tag ",comments".  The struct
//      field may have type []byte or string.  If there is no such
//      field, the comments are discarded.
//
//   * If the XML element contains a sub-element whose name matches
//      the prefix of a tag formatted as "a" or "a>b>c", unmarshal
//      will descend into the XML structure looking for elements with the
//      given names, and will map the innermost elements to that struct
//      field. A tag starting with ">" is equivalent to one starting
//      with the field name followed by ">".
//
//   * If the XML element contains a sub-element whose name matches
//      a struct field's XMLName tag and the struct field has no
//      explicit name tag as per the previous rule, unmarshal maps
//      the sub-element to that struct field.
//
//   * If the XML element contains a sub-element whose name matches a
//      field without any mode flags (",attr", ",chardata", etc), Unmarshal
//      maps the sub-element to that struct field.
//
//   * If the XML element contains a sub-element that hasn't matched any
//      of the above rules and the struct has a field with tag ",any",
//      unmarshal maps the sub-element to that struct field.
//
// Unmarshal maps an XML element to a string or []byte by saving the
// concatenation of that element's character data in the string or
// []byte.
//
// Unmarshal maps an attribute value to a string or []byte by saving
// the value in the string or slice.
//
// Unmarshal maps an XML element to a slice by extending the length of
// the slice and mapping the element to the newly created value.
//
// Unmarshal maps an XML element or attribute value to a bool by
// setting it to the boolean value represented by the string.
//
// Unmarshal maps an XML element or attribute value to an integer or
// floating-point field by setting the field to the result of
// interpreting the string value in decimal.  There is no check for
// overflow.
//
// Unmarshal maps an XML element to an xml.Name by recording the
// element name.
//
// Unmarshal maps an XML element to a pointer by setting the pointer
// to a freshly allocated value and then mapping the element to that value.
//
func Unmarshal(r io.Reader, val interface{}) error {
	v := reflect.ValueOf(val)
	if v.Kind() != reflect.Ptr {
		return errors.New("non-pointer passed to Unmarshal")
	}
	p := NewParser(r)
	elem := v.Elem()
	err := p.unmarshal(elem, nil)
	if err != nil {
		return err
	}
	return nil
}

// An UnmarshalError represents an error in the unmarshalling process.
type UnmarshalError string

func (e UnmarshalError) Error() string { return string(e) }

// The Parser's Unmarshal method is like xml.Unmarshal
// except that it can be passed a pointer to the initial start element,
// useful when a client reads some raw XML tokens itself
// but also defers to Unmarshal for some elements.
// Passing a nil start element indicates that Unmarshal should
// read the token stream to find the start element.
func (p *Parser) Unmarshal(val interface{}, start *StartElement) error {
	v := reflect.ValueOf(val)
	if v.Kind() != reflect.Ptr {
		return errors.New("non-pointer passed to Unmarshal")
	}
	return p.unmarshal(v.Elem(), start)
}

// Unmarshal a single XML element into val.
func (p *Parser) unmarshal(val reflect.Value, start *StartElement) error {
	// Find start element if we need it.
	if start == nil {
		for {
			tok, err := p.Token()
			if err != nil {
				return err
			}
			if t, ok := tok.(StartElement); ok {
				start = &t
				break
			}
		}
	}

	if pv := val; pv.Kind() == reflect.Ptr {
		if pv.IsNil() {
			pv.Set(reflect.New(pv.Type().Elem()))
		}
		val = pv.Elem()
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
		return p.Skip()

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
		if n >= v.Cap() {
			ncap := 2 * n
			if ncap < 4 {
				ncap = 4
			}
			new := reflect.MakeSlice(typ, n, ncap)
			reflect.Copy(new, v)
			v.Set(new)
		}
		v.SetLen(n + 1)

		// Recur to read element into slice.
		if err := p.unmarshal(v.Index(n), start); err != nil {
			v.SetLen(n)
			return err
		}
		return nil

	case reflect.Bool, reflect.Float32, reflect.Float64, reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64, reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr, reflect.String:
		saveData = v

	case reflect.Struct:
		sv = v
		typ := sv.Type()
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
			fv := sv.FieldByIndex(finfo.idx)
			if _, ok := fv.Interface().(Name); ok {
				fv.Set(reflect.ValueOf(start.Name))
			}
		}

		// Assign attributes.
		// Also, determine whether we need to save character data or comments.
		for i := range tinfo.fields {
			finfo := &tinfo.fields[i]
			switch finfo.flags & fMode {
			case fAttr:
				strv := sv.FieldByIndex(finfo.idx)
				// Look for attribute.
				val := ""
				for _, a := range start.Attr {
					if a.Name.Local == finfo.name {
						val = a.Value
						break
					}
				}
				copyValue(strv, []byte(val))

			case fCharData:
				if !saveData.IsValid() {
					saveData = sv.FieldByIndex(finfo.idx)
				}

			case fComment:
				if !saveComment.IsValid() {
					saveComment = sv.FieldByIndex(finfo.idx)
				}

			case fAny:
				if !saveAny.IsValid() {
					saveAny = sv.FieldByIndex(finfo.idx)
				}

			case fInnerXml:
				if !saveXML.IsValid() {
					saveXML = sv.FieldByIndex(finfo.idx)
					if p.saved == nil {
						saveXMLIndex = 0
						p.saved = new(bytes.Buffer)
					} else {
						saveXMLIndex = p.savedOffset()
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
			savedOffset = p.savedOffset()
		}
		tok, err := p.Token()
		if err != nil {
			return err
		}
		switch t := tok.(type) {
		case StartElement:
			consumed := false
			if sv.IsValid() {
				consumed, err = p.unmarshalPath(tinfo, sv, nil, &t)
				if err != nil {
					return err
				}
				if !consumed && saveAny.IsValid() {
					consumed = true
					if err := p.unmarshal(saveAny, &t); err != nil {
						return err
					}
				}
			}
			if !consumed {
				if err := p.Skip(); err != nil {
					return err
				}
			}

		case EndElement:
			if saveXML.IsValid() {
				saveXMLData = p.saved.Bytes()[saveXMLIndex:savedOffset]
				if saveXMLIndex == 0 {
					p.saved = nil
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
		t.Set(reflect.ValueOf(saveXMLData))
	}

	return nil
}

func copyValue(dst reflect.Value, src []byte) (err error) {
	// Helper functions for integer and unsigned integer conversions
	var itmp int64
	getInt64 := func() bool {
		itmp, err = strconv.ParseInt(string(src), 10, 64)
		// TODO: should check sizes
		return err == nil
	}
	var utmp uint64
	getUint64 := func() bool {
		utmp, err = strconv.ParseUint(string(src), 10, 64)
		// TODO: check for overflow?
		return err == nil
	}
	var ftmp float64
	getFloat64 := func() bool {
		ftmp, err = strconv.ParseFloat(string(src), 64)
		// TODO: check for overflow?
		return err == nil
	}

	// Save accumulated data.
	switch t := dst; t.Kind() {
	case reflect.Invalid:
		// Probably a comment.
	default:
		return errors.New("cannot happen: unknown type " + t.Type().String())
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		if !getInt64() {
			return err
		}
		t.SetInt(itmp)
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		if !getUint64() {
			return err
		}
		t.SetUint(utmp)
	case reflect.Float32, reflect.Float64:
		if !getFloat64() {
			return err
		}
		t.SetFloat(ftmp)
	case reflect.Bool:
		value, err := strconv.ParseBool(strings.TrimSpace(string(src)))
		if err != nil {
			return err
		}
		t.SetBool(value)
	case reflect.String:
		t.SetString(string(src))
	case reflect.Slice:
		t.Set(reflect.ValueOf(src))
	}
	return nil
}

// unmarshalPath walks down an XML structure looking for wanted
// paths, and calls unmarshal on them.
// The consumed result tells whether XML elements have been consumed
// from the Parser until start's matching end element, or if it's
// still untouched because start is uninteresting for sv's fields.
func (p *Parser) unmarshalPath(tinfo *typeInfo, sv reflect.Value, parents []string, start *StartElement) (consumed bool, err error) {
	recurse := false
Loop:
	for i := range tinfo.fields {
		finfo := &tinfo.fields[i]
		if finfo.flags&fElement == 0 || len(finfo.parents) < len(parents) {
			continue
		}
		for j := range parents {
			if parents[j] != finfo.parents[j] {
				continue Loop
			}
		}
		if len(finfo.parents) == len(parents) && finfo.name == start.Name.Local {
			// It's a perfect match, unmarshal the field.
			return true, p.unmarshal(sv.FieldByIndex(finfo.idx), start)
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
		tok, err = p.Token()
		if err != nil {
			return true, err
		}
		switch t := tok.(type) {
		case StartElement:
			consumed2, err := p.unmarshalPath(tinfo, sv, parents, &t)
			if err != nil {
				return true, err
			}
			if !consumed2 {
				if err := p.Skip(); err != nil {
					return true, err
				}
			}
		case EndElement:
			return true, nil
		}
	}
	panic("unreachable")
}

// Have already read a start element.
// Read tokens until we find the end element.
// Token is taking care of making sure the
// end element matches the start element we saw.
func (p *Parser) Skip() error {
	for {
		tok, err := p.Token()
		if err != nil {
			return err
		}
		switch tok.(type) {
		case StartElement:
			if err := p.Skip(); err != nil {
				return err
			}
		case EndElement:
			return nil
		}
	}
	panic("unreachable")
}
