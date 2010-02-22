// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package xml

import (
	"bytes"
	"io"
	"os"
	"reflect"
	"strconv"
	"strings"
	"unicode"
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
//		Where string "attr";
//		Addr string;
//	}
//
//	type Result struct {
//		XMLName xml.Name "result";
//		Name string;
//		Phone string;
//		Email []Email;
//	}
//
//	result := Result{ Name: "name", Phone: "phone", Email: nil }
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
//		<address>123 Main Street</address>
//	</result>
//
// via Unmarshal(r, &result) is equivalent to assigning
//
//	r = Result{
//		xml.Name{"", "result"},
//		"Grace R. Emlin",	// name
//		"phone",	// no phone given
//		[]Email{
//			Email{ "home", "gre@example.com" },
//			Email{ "work", "gre@work.com" }
//		}
//	}
//
// Note that the field r.Phone has not been modified and
// that the XML <address> element was discarded.
//
// Because Unmarshal uses the reflect package, it can only
// assign to upper case fields.  Unmarshal uses a case-insensitive
// comparison to match XML element names to struct field names.
//
// Unmarshal maps an XML element to a struct using the following rules:
//
//   * If the struct has a field named XMLName of type xml.Name,
//      Unmarshal records the element name in that field.
//
//   * If the XMLName field has an associated tag string of the form
//      "tag" or "namespace-URL tag", the XML element must have
//      the given tag (and, optionally, name space) or else Unmarshal
//      returns an error.
//
//   * If the XML element has an attribute whose name matches a
//      struct field of type string with tag "attr", Unmarshal records
//      the attribute value in that field.
//
//   * If the XML element contains character data, that data is
//      accumulated in the first struct field that has tag "chardata".
//      The struct field may have type []byte or string.
//      If there is no such field, the character data is discarded.
//
//   * If the XML element contains a sub-element whose name
//      matches a struct field whose tag is neither "attr" nor "chardata",
//      Unmarshal maps the sub-element to that struct field.
//      Otherwise, if the struct has a field named Any, unmarshal
//      maps the sub-element to that struct field.
//
// Unmarshal maps an XML element to a string or []byte by saving the
// concatenation of that elements character data in the string or []byte.
//
// Unmarshal maps an XML element to a slice by extending the length
// of the slice and mapping the element to the newly created value.
//
// Unmarshal maps an XML element to a bool by setting it true if the
// string value is "true" or "1", or false otherwise.
//
// Unmarshal maps an XML element to an integer or floating-point
// field by setting the field to the result of interpreting the string
// value in decimal.  There is no check for overflow.
//
// Unmarshal maps an XML element to an xml.Name by recording the
// element name.
//
// Unmarshal maps an XML element to a pointer by setting the pointer
// to a freshly allocated value and then mapping the element to that value.
//
func Unmarshal(r io.Reader, val interface{}) os.Error {
	v, ok := reflect.NewValue(val).(*reflect.PtrValue)
	if !ok {
		return os.NewError("non-pointer passed to Unmarshal")
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

func (e UnmarshalError) String() string { return string(e) }

// The Parser's Unmarshal method is like xml.Unmarshal
// except that it can be passed a pointer to the initial start element,
// useful when a client reads some raw XML tokens itself
// but also defers to Unmarshal for some elements.
// Passing a nil start element indicates that Unmarshal should
// read the token stream to find the start element.
func (p *Parser) Unmarshal(val interface{}, start *StartElement) os.Error {
	v, ok := reflect.NewValue(val).(*reflect.PtrValue)
	if !ok {
		return os.NewError("non-pointer passed to Unmarshal")
	}
	return p.unmarshal(v.Elem(), start)
}

// fieldName strips invalid characters from an XML name
// to create a valid Go struct name.  It also converts the
// name to lower case letters.
func fieldName(original string) string {

	var i int
	//remove leading underscores
	for i = 0; i < len(original) && original[i] == '_'; i++ {
	}

	return strings.Map(
		func(x int) int {
			if x == '_' || unicode.IsDigit(x) || unicode.IsLetter(x) {
				return unicode.ToLower(x)
			}
			return -1
		},
		original[i:])
}

// Unmarshal a single XML element into val.
func (p *Parser) unmarshal(val reflect.Value, start *StartElement) os.Error {
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

	if pv, ok := val.(*reflect.PtrValue); ok {
		if pv.Get() == 0 {
			zv := reflect.MakeZero(pv.Type().(*reflect.PtrType).Elem())
			pv.PointTo(zv)
			val = zv
		} else {
			val = pv.Elem()
		}
	}

	var (
		data        []byte
		saveData    reflect.Value
		comment     []byte
		saveComment reflect.Value
		sv          *reflect.StructValue
		styp        *reflect.StructType
	)
	switch v := val.(type) {
	default:
		return os.ErrorString("unknown type " + v.Type().String())

	case *reflect.SliceValue:
		typ := v.Type().(*reflect.SliceType)
		if _, ok := typ.Elem().(*reflect.Uint8Type); ok {
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
			reflect.ArrayCopy(new, v)
			v.Set(new)
		}
		v.SetLen(n + 1)

		// Recur to read element into slice.
		if err := p.unmarshal(v.Elem(n), start); err != nil {
			v.SetLen(n)
			return err
		}
		return nil

	case *reflect.StringValue,
		*reflect.IntValue, *reflect.UintValue, *reflect.UintptrValue,
		*reflect.Int8Value, *reflect.Int16Value, *reflect.Int32Value, *reflect.Int64Value,
		*reflect.Uint8Value, *reflect.Uint16Value, *reflect.Uint32Value, *reflect.Uint64Value,
		*reflect.FloatValue, *reflect.Float32Value, *reflect.Float64Value, *reflect.BoolValue:
		saveData = v

	case *reflect.StructValue:
		if _, ok := v.Interface().(Name); ok {
			v.Set(reflect.NewValue(start.Name).(*reflect.StructValue))
			break
		}

		sv = v
		typ := sv.Type().(*reflect.StructType)
		styp = typ
		// Assign name.
		if f, ok := typ.FieldByName("XMLName"); ok {
			// Validate element name.
			if f.Tag != "" {
				tag := f.Tag
				ns := ""
				i := strings.LastIndex(tag, " ")
				if i >= 0 {
					ns, tag = tag[0:i], tag[i+1:]
				}
				if tag != start.Name.Local {
					return UnmarshalError("expected element type <" + tag + "> but have <" + start.Name.Local + ">")
				}
				if ns != "" && ns != start.Name.Space {
					e := "expected element <" + tag + "> in name space " + ns + " but have "
					if start.Name.Space == "" {
						e += "no name space"
					} else {
						e += start.Name.Space
					}
					return UnmarshalError(e)
				}
			}

			// Save
			v := sv.FieldByIndex(f.Index)
			if _, ok := v.Interface().(Name); !ok {
				return UnmarshalError(sv.Type().String() + " field XMLName does not have type xml.Name")
			}
			v.(*reflect.StructValue).Set(reflect.NewValue(start.Name).(*reflect.StructValue))
		}

		// Assign attributes.
		// Also, determine whether we need to save character data or comments.
		for i, n := 0, typ.NumField(); i < n; i++ {
			f := typ.Field(i)
			switch f.Tag {
			case "attr":
				strv, ok := sv.FieldByIndex(f.Index).(*reflect.StringValue)
				if !ok {
					return UnmarshalError(sv.Type().String() + " field " + f.Name + " has attr tag but is not type string")
				}
				// Look for attribute.
				val := ""
				k := strings.ToLower(f.Name)
				for _, a := range start.Attr {
					if fieldName(a.Name.Local) == k {
						val = a.Value
						break
					}
				}
				strv.Set(val)

			case "comment":
				if saveComment == nil {
					saveComment = sv.FieldByIndex(f.Index)
				}

			case "chardata":
				if saveData == nil {
					saveData = sv.FieldByIndex(f.Index)
				}
			}
		}
	}

	// Find end element.
	// Process sub-elements along the way.
Loop:
	for {
		tok, err := p.Token()
		if err != nil {
			return err
		}
		switch t := tok.(type) {
		case StartElement:
			// Sub-element.
			// Look up by tag name.
			// If that fails, fall back to mop-up field named "Any".
			if sv != nil {
				k := fieldName(t.Name.Local)
				any := -1
				for i, n := 0, styp.NumField(); i < n; i++ {
					f := styp.Field(i)
					if strings.ToLower(f.Name) == k {
						if err := p.unmarshal(sv.FieldByIndex(f.Index), &t); err != nil {
							return err
						}
						continue Loop
					}
					if any < 0 && f.Name == "Any" {
						any = i
					}
				}
				if any >= 0 {
					if err := p.unmarshal(sv.FieldByIndex(styp.Field(any).Index), &t); err != nil {
						return err
					}
					continue Loop
				}
			}
			// Not saving sub-element but still have to skip over it.
			if err := p.Skip(); err != nil {
				return err
			}

		case EndElement:
			break Loop

		case CharData:
			if saveData != nil {
				data = bytes.Add(data, t)
			}

		case Comment:
			if saveComment != nil {
				comment = bytes.Add(comment, t)
			}
		}
	}

	var err os.Error
	// Helper functions for integer and unsigned integer conversions
	var itmp int64
	getInt64 := func() bool {
		itmp, err = strconv.Atoi64(string(data))
		// TODO: should check sizes
		return err == nil
	}
	var utmp uint64
	getUint64 := func() bool {
		utmp, err = strconv.Atoui64(string(data))
		// TODO: check for overflow?
		return err == nil
	}
	var ftmp float64
	getFloat64 := func() bool {
		ftmp, err = strconv.Atof64(string(data))
		// TODO: check for overflow?
		return err == nil
	}

	// Save accumulated data and comments
	switch t := saveData.(type) {
	case nil:
		// Probably a comment, handled below
	default:
		return os.ErrorString("cannot happen: unknown type " + t.Type().String())
	case *reflect.IntValue:
		if !getInt64() {
			return err
		}
		t.Set(int(itmp))
	case *reflect.Int8Value:
		if !getInt64() {
			return err
		}
		t.Set(int8(itmp))
	case *reflect.Int16Value:
		if !getInt64() {
			return err
		}
		t.Set(int16(itmp))
	case *reflect.Int32Value:
		if !getInt64() {
			return err
		}
		t.Set(int32(itmp))
	case *reflect.Int64Value:
		if !getInt64() {
			return err
		}
		t.Set(itmp)
	case *reflect.UintValue:
		if !getUint64() {
			return err
		}
		t.Set(uint(utmp))
	case *reflect.Uint8Value:
		if !getUint64() {
			return err
		}
		t.Set(uint8(utmp))
	case *reflect.Uint16Value:
		if !getUint64() {
			return err
		}
		t.Set(uint16(utmp))
	case *reflect.Uint32Value:
		if !getUint64() {
			return err
		}
		t.Set(uint32(utmp))
	case *reflect.Uint64Value:
		if !getUint64() {
			return err
		}
		t.Set(utmp)
	case *reflect.UintptrValue:
		if !getUint64() {
			return err
		}
		t.Set(uintptr(utmp))
	case *reflect.FloatValue:
		if !getFloat64() {
			return err
		}
		t.Set(float(ftmp))
	case *reflect.Float32Value:
		if !getFloat64() {
			return err
		}
		t.Set(float32(ftmp))
	case *reflect.Float64Value:
		if !getFloat64() {
			return err
		}
		t.Set(ftmp)
	case *reflect.BoolValue:
		btmp := strings.TrimSpace(string(data))
		t.Set(strings.ToLower(btmp) == "true" || btmp == "1")
	case *reflect.StringValue:
		t.Set(string(data))
	case *reflect.SliceValue:
		t.Set(reflect.NewValue(data).(*reflect.SliceValue))
	}

	switch t := saveComment.(type) {
	case *reflect.StringValue:
		t.Set(string(comment))
	case *reflect.SliceValue:
		t.Set(reflect.NewValue(comment).(*reflect.SliceValue))
	}

	return nil
}

// Have already read a start element.
// Read tokens until we find the end element.
// Token is taking care of making sure the
// end element matches the start element we saw.
func (p *Parser) Skip() os.Error {
	for {
		tok, err := p.Token()
		if err != nil {
			return err
		}
		switch t := tok.(type) {
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
