// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Represents JSON data structure using native Go types: booleans, floats,
// strings, arrays, and maps.

package json

import (
	"encoding/base64"
	"errors"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf16"
	"unicode/utf8"
)

// Unmarshal parses the JSON-encoded data and stores the result
// in the value pointed to by v.
//
// Unmarshal uses the inverse of the encodings that
// Marshal uses, allocating maps, slices, and pointers as necessary,
// with the following additional rules:
//
// To unmarshal JSON into a pointer, Unmarshal first handles the case of
// the JSON being the JSON literal null.  In that case, Unmarshal sets
// the pointer to nil.  Otherwise, Unmarshal unmarshals the JSON into
// the value pointed at by the pointer.  If the pointer is nil, Unmarshal
// allocates a new value for it to point to.
//
// To unmarshal JSON into an interface value, Unmarshal unmarshals
// the JSON into the concrete value contained in the interface value.
// If the interface value is nil, that is, has no concrete value stored in it,
// Unmarshal stores one of these in the interface value:
//
//	bool, for JSON booleans
//	float64, for JSON numbers
//	string, for JSON strings
//	[]interface{}, for JSON arrays
//	map[string]interface{}, for JSON objects
//	nil for JSON null
//
// If a JSON value is not appropriate for a given target type,
// or if a JSON number overflows the target type, Unmarshal
// skips that field and completes the unmarshalling as best it can.
// If no more serious errors are encountered, Unmarshal returns
// an UnmarshalTypeError describing the earliest such error.
//
func Unmarshal(data []byte, v interface{}) error {
	d := new(decodeState).init(data)

	// Quick check for well-formedness.
	// Avoids filling out half a data structure
	// before discovering a JSON syntax error.
	err := checkValid(data, &d.scan)
	if err != nil {
		return err
	}

	return d.unmarshal(v)
}

// Unmarshaler is the interface implemented by objects
// that can unmarshal a JSON description of themselves.
// The input can be assumed to be a valid JSON object
// encoding.  UnmarshalJSON must copy the JSON data
// if it wishes to retain the data after returning.
type Unmarshaler interface {
	UnmarshalJSON([]byte) error
}

// An UnmarshalTypeError describes a JSON value that was
// not appropriate for a value of a specific Go type.
type UnmarshalTypeError struct {
	Value string       // description of JSON value - "bool", "array", "number -5"
	Type  reflect.Type // type of Go value it could not be assigned to
}

func (e *UnmarshalTypeError) Error() string {
	return "json: cannot unmarshal " + e.Value + " into Go value of type " + e.Type.String()
}

// An UnmarshalFieldError describes a JSON object key that
// led to an unexported (and therefore unwritable) struct field.
type UnmarshalFieldError struct {
	Key   string
	Type  reflect.Type
	Field reflect.StructField
}

func (e *UnmarshalFieldError) Error() string {
	return "json: cannot unmarshal object key " + strconv.Quote(e.Key) + " into unexported field " + e.Field.Name + " of type " + e.Type.String()
}

// An InvalidUnmarshalError describes an invalid argument passed to Unmarshal.
// (The argument to Unmarshal must be a non-nil pointer.)
type InvalidUnmarshalError struct {
	Type reflect.Type
}

func (e *InvalidUnmarshalError) Error() string {
	if e.Type == nil {
		return "json: Unmarshal(nil)"
	}

	if e.Type.Kind() != reflect.Ptr {
		return "json: Unmarshal(non-pointer " + e.Type.String() + ")"
	}
	return "json: Unmarshal(nil " + e.Type.String() + ")"
}

func (d *decodeState) unmarshal(v interface{}) (err error) {
	defer func() {
		if r := recover(); r != nil {
			if _, ok := r.(runtime.Error); ok {
				panic(r)
			}
			err = r.(error)
		}
	}()

	rv := reflect.ValueOf(v)
	pv := rv
	if pv.Kind() != reflect.Ptr || pv.IsNil() {
		return &InvalidUnmarshalError{reflect.TypeOf(v)}
	}

	d.scan.reset()
	// We decode rv not pv.Elem because the Unmarshaler interface
	// test must be applied at the top level of the value.
	d.value(rv)
	return d.savedError
}

// decodeState represents the state while decoding a JSON value.
type decodeState struct {
	data       []byte
	off        int // read offset in data
	scan       scanner
	nextscan   scanner // for calls to nextValue
	savedError error
	tempstr    string // scratch space to avoid some allocations
}

// errPhase is used for errors that should not happen unless
// there is a bug in the JSON decoder or something is editing
// the data slice while the decoder executes.
var errPhase = errors.New("JSON decoder out of sync - data changing underfoot?")

func (d *decodeState) init(data []byte) *decodeState {
	d.data = data
	d.off = 0
	d.savedError = nil
	return d
}

// error aborts the decoding by panicking with err.
func (d *decodeState) error(err error) {
	panic(err)
}

// saveError saves the first err it is called with,
// for reporting at the end of the unmarshal.
func (d *decodeState) saveError(err error) {
	if d.savedError == nil {
		d.savedError = err
	}
}

// next cuts off and returns the next full JSON value in d.data[d.off:].
// The next value is known to be an object or array, not a literal.
func (d *decodeState) next() []byte {
	c := d.data[d.off]
	item, rest, err := nextValue(d.data[d.off:], &d.nextscan)
	if err != nil {
		d.error(err)
	}
	d.off = len(d.data) - len(rest)

	// Our scanner has seen the opening brace/bracket
	// and thinks we're still in the middle of the object.
	// invent a closing brace/bracket to get it out.
	if c == '{' {
		d.scan.step(&d.scan, '}')
	} else {
		d.scan.step(&d.scan, ']')
	}

	return item
}

// scanWhile processes bytes in d.data[d.off:] until it
// receives a scan code not equal to op.
// It updates d.off and returns the new scan code.
func (d *decodeState) scanWhile(op int) int {
	var newOp int
	for {
		if d.off >= len(d.data) {
			newOp = d.scan.eof()
			d.off = len(d.data) + 1 // mark processed EOF with len+1
		} else {
			c := int(d.data[d.off])
			d.off++
			newOp = d.scan.step(&d.scan, c)
		}
		if newOp != op {
			break
		}
	}
	return newOp
}

// value decodes a JSON value from d.data[d.off:] into the value.
// it updates d.off to point past the decoded value.
func (d *decodeState) value(v reflect.Value) {
	if !v.IsValid() {
		_, rest, err := nextValue(d.data[d.off:], &d.nextscan)
		if err != nil {
			d.error(err)
		}
		d.off = len(d.data) - len(rest)

		// d.scan thinks we're still at the beginning of the item.
		// Feed in an empty string - the shortest, simplest value -
		// so that it knows we got to the end of the value.
		if d.scan.redo {
			// rewind.
			d.scan.redo = false
			d.scan.step = stateBeginValue
		}
		d.scan.step(&d.scan, '"')
		d.scan.step(&d.scan, '"')
		return
	}

	switch op := d.scanWhile(scanSkipSpace); op {
	default:
		d.error(errPhase)

	case scanBeginArray:
		d.array(v)

	case scanBeginObject:
		d.object(v)

	case scanBeginLiteral:
		d.literal(v)
	}
}

// indirect walks down v allocating pointers as needed,
// until it gets to a non-pointer.
// if it encounters an Unmarshaler, indirect stops and returns that.
// if decodingNull is true, indirect stops at the last pointer so it can be set to nil.
func (d *decodeState) indirect(v reflect.Value, decodingNull bool) (Unmarshaler, reflect.Value) {
	// If v is a named type and is addressable,
	// start with its address, so that if the type has pointer methods,
	// we find them.
	if v.Kind() != reflect.Ptr && v.Type().Name() != "" && v.CanAddr() {
		v = v.Addr()
	}
	for {
		var isUnmarshaler bool
		if v.Type().NumMethod() > 0 {
			// Remember that this is an unmarshaler,
			// but wait to return it until after allocating
			// the pointer (if necessary).
			_, isUnmarshaler = v.Interface().(Unmarshaler)
		}

		if iv := v; iv.Kind() == reflect.Interface && !iv.IsNil() {
			v = iv.Elem()
			continue
		}

		pv := v
		if pv.Kind() != reflect.Ptr {
			break
		}

		if pv.Elem().Kind() != reflect.Ptr && decodingNull && pv.CanSet() {
			return nil, pv
		}
		if pv.IsNil() {
			pv.Set(reflect.New(pv.Type().Elem()))
		}
		if isUnmarshaler {
			// Using v.Interface().(Unmarshaler)
			// here means that we have to use a pointer
			// as the struct field.  We cannot use a value inside
			// a pointer to a struct, because in that case
			// v.Interface() is the value (x.f) not the pointer (&x.f).
			// This is an unfortunate consequence of reflect.
			// An alternative would be to look up the
			// UnmarshalJSON method and return a FuncValue.
			return v.Interface().(Unmarshaler), reflect.Value{}
		}
		v = pv.Elem()
	}
	return nil, v
}

// array consumes an array from d.data[d.off-1:], decoding into the value v.
// the first byte of the array ('[') has been read already.
func (d *decodeState) array(v reflect.Value) {
	// Check for unmarshaler.
	unmarshaler, pv := d.indirect(v, false)
	if unmarshaler != nil {
		d.off--
		err := unmarshaler.UnmarshalJSON(d.next())
		if err != nil {
			d.error(err)
		}
		return
	}
	v = pv

	// Check type of target.
	switch v.Kind() {
	default:
		d.saveError(&UnmarshalTypeError{"array", v.Type()})
		d.off--
		d.next()
		return
	case reflect.Interface:
		// Decoding into nil interface?  Switch to non-reflect code.
		v.Set(reflect.ValueOf(d.arrayInterface()))
		return
	case reflect.Array:
	case reflect.Slice:
		break
	}

	i := 0
	for {
		// Look ahead for ] - can only happen on first iteration.
		op := d.scanWhile(scanSkipSpace)
		if op == scanEndArray {
			break
		}

		// Back up so d.value can have the byte we just read.
		d.off--
		d.scan.undo(op)

		// Get element of array, growing if necessary.
		if v.Kind() == reflect.Slice {
			// Grow slice if necessary
			if i >= v.Cap() {
				newcap := v.Cap() + v.Cap()/2
				if newcap < 4 {
					newcap = 4
				}
				newv := reflect.MakeSlice(v.Type(), v.Len(), newcap)
				reflect.Copy(newv, v)
				v.Set(newv)
			}
			if i >= v.Len() {
				v.SetLen(i + 1)
			}
		}

		if i < v.Len() {
			// Decode into element.
			d.value(v.Index(i))
		} else {
			// Ran out of fixed array: skip.
			d.value(reflect.Value{})
		}
		i++

		// Next token must be , or ].
		op = d.scanWhile(scanSkipSpace)
		if op == scanEndArray {
			break
		}
		if op != scanArrayValue {
			d.error(errPhase)
		}
	}

	if i < v.Len() {
		if v.Kind() == reflect.Array {
			// Array.  Zero the rest.
			z := reflect.Zero(v.Type().Elem())
			for ; i < v.Len(); i++ {
				v.Index(i).Set(z)
			}
		} else {
			v.SetLen(i)
		}
	}
	if i == 0 && v.Kind() == reflect.Slice {
		v.Set(reflect.MakeSlice(v.Type(), 0, 0))
	}
}

// object consumes an object from d.data[d.off-1:], decoding into the value v.
// the first byte of the object ('{') has been read already.
func (d *decodeState) object(v reflect.Value) {
	// Check for unmarshaler.
	unmarshaler, pv := d.indirect(v, false)
	if unmarshaler != nil {
		d.off--
		err := unmarshaler.UnmarshalJSON(d.next())
		if err != nil {
			d.error(err)
		}
		return
	}
	v = pv

	// Decoding into nil interface?  Switch to non-reflect code.
	iv := v
	if iv.Kind() == reflect.Interface {
		iv.Set(reflect.ValueOf(d.objectInterface()))
		return
	}

	// Check type of target: struct or map[string]T
	var (
		mv reflect.Value
		sv reflect.Value
	)
	switch v.Kind() {
	case reflect.Map:
		// map must have string type
		t := v.Type()
		if t.Key() != reflect.TypeOf("") {
			d.saveError(&UnmarshalTypeError{"object", v.Type()})
			break
		}
		mv = v
		if mv.IsNil() {
			mv.Set(reflect.MakeMap(t))
		}
	case reflect.Struct:
		sv = v
	default:
		d.saveError(&UnmarshalTypeError{"object", v.Type()})
	}

	if !mv.IsValid() && !sv.IsValid() {
		d.off--
		d.next() // skip over { } in input
		return
	}

	var mapElem reflect.Value

	for {
		// Read opening " of string key or closing }.
		op := d.scanWhile(scanSkipSpace)
		if op == scanEndObject {
			// closing } - can only happen on first iteration.
			break
		}
		if op != scanBeginLiteral {
			d.error(errPhase)
		}

		// Read string key.
		start := d.off - 1
		op = d.scanWhile(scanContinue)
		item := d.data[start : d.off-1]
		key, ok := unquote(item)
		if !ok {
			d.error(errPhase)
		}

		// Figure out field corresponding to key.
		var subv reflect.Value
		destring := false // whether the value is wrapped in a string to be decoded first

		if mv.IsValid() {
			elemType := mv.Type().Elem()
			if !mapElem.IsValid() {
				mapElem = reflect.New(elemType).Elem()
			} else {
				mapElem.Set(reflect.Zero(elemType))
			}
			subv = mapElem
		} else {
			var f reflect.StructField
			var ok bool
			st := sv.Type()
			for i := 0; i < sv.NumField(); i++ {
				sf := st.Field(i)
				tag := sf.Tag.Get("json")
				if tag == "-" {
					// Pretend this field doesn't exist.
					continue
				}
				// First, tag match
				tagName, _ := parseTag(tag)
				if tagName == key {
					f = sf
					ok = true
					break // no better match possible
				}
				// Second, exact field name match
				if sf.Name == key {
					f = sf
					ok = true
				}
				// Third, case-insensitive field name match,
				// but only if a better match hasn't already been seen
				if !ok && strings.EqualFold(sf.Name, key) {
					f = sf
					ok = true
				}
			}

			// Extract value; name must be exported.
			if ok {
				if f.PkgPath != "" {
					d.saveError(&UnmarshalFieldError{key, st, f})
				} else {
					subv = sv.FieldByIndex(f.Index)
				}
				_, opts := parseTag(f.Tag.Get("json"))
				destring = opts.Contains("string")
			}
		}

		// Read : before value.
		if op == scanSkipSpace {
			op = d.scanWhile(scanSkipSpace)
		}
		if op != scanObjectKey {
			d.error(errPhase)
		}

		// Read value.
		if destring {
			d.value(reflect.ValueOf(&d.tempstr))
			d.literalStore([]byte(d.tempstr), subv)
		} else {
			d.value(subv)
		}
		// Write value back to map;
		// if using struct, subv points into struct already.
		if mv.IsValid() {
			mv.SetMapIndex(reflect.ValueOf(key), subv)
		}

		// Next token must be , or }.
		op = d.scanWhile(scanSkipSpace)
		if op == scanEndObject {
			break
		}
		if op != scanObjectValue {
			d.error(errPhase)
		}
	}
}

// literal consumes a literal from d.data[d.off-1:], decoding into the value v.
// The first byte of the literal has been read already
// (that's how the caller knows it's a literal).
func (d *decodeState) literal(v reflect.Value) {
	// All bytes inside literal return scanContinue op code.
	start := d.off - 1
	op := d.scanWhile(scanContinue)

	// Scan read one byte too far; back up.
	d.off--
	d.scan.undo(op)

	d.literalStore(d.data[start:d.off], v)
}

// literalStore decodes a literal stored in item into v.
func (d *decodeState) literalStore(item []byte, v reflect.Value) {
	// Check for unmarshaler.
	wantptr := item[0] == 'n' // null
	unmarshaler, pv := d.indirect(v, wantptr)
	if unmarshaler != nil {
		err := unmarshaler.UnmarshalJSON(item)
		if err != nil {
			d.error(err)
		}
		return
	}
	v = pv

	switch c := item[0]; c {
	case 'n': // null
		switch v.Kind() {
		default:
			d.saveError(&UnmarshalTypeError{"null", v.Type()})
		case reflect.Interface, reflect.Ptr, reflect.Map, reflect.Slice:
			v.Set(reflect.Zero(v.Type()))
		}

	case 't', 'f': // true, false
		value := c == 't'
		switch v.Kind() {
		default:
			d.saveError(&UnmarshalTypeError{"bool", v.Type()})
		case reflect.Bool:
			v.SetBool(value)
		case reflect.Interface:
			v.Set(reflect.ValueOf(value))
		}

	case '"': // string
		s, ok := unquoteBytes(item)
		if !ok {
			d.error(errPhase)
		}
		switch v.Kind() {
		default:
			d.saveError(&UnmarshalTypeError{"string", v.Type()})
		case reflect.Slice:
			if v.Type() != byteSliceType {
				d.saveError(&UnmarshalTypeError{"string", v.Type()})
				break
			}
			b := make([]byte, base64.StdEncoding.DecodedLen(len(s)))
			n, err := base64.StdEncoding.Decode(b, s)
			if err != nil {
				d.saveError(err)
				break
			}
			v.Set(reflect.ValueOf(b[0:n]))
		case reflect.String:
			v.SetString(string(s))
		case reflect.Interface:
			v.Set(reflect.ValueOf(string(s)))
		}

	default: // number
		if c != '-' && (c < '0' || c > '9') {
			d.error(errPhase)
		}
		s := string(item)
		switch v.Kind() {
		default:
			d.error(&UnmarshalTypeError{"number", v.Type()})
		case reflect.Interface:
			n, err := strconv.ParseFloat(s, 64)
			if err != nil {
				d.saveError(&UnmarshalTypeError{"number " + s, v.Type()})
				break
			}
			v.Set(reflect.ValueOf(n))

		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			n, err := strconv.ParseInt(s, 10, 64)
			if err != nil || v.OverflowInt(n) {
				d.saveError(&UnmarshalTypeError{"number " + s, v.Type()})
				break
			}
			v.SetInt(n)

		case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
			n, err := strconv.ParseUint(s, 10, 64)
			if err != nil || v.OverflowUint(n) {
				d.saveError(&UnmarshalTypeError{"number " + s, v.Type()})
				break
			}
			v.SetUint(n)

		case reflect.Float32, reflect.Float64:
			n, err := strconv.ParseFloat(s, v.Type().Bits())
			if err != nil || v.OverflowFloat(n) {
				d.saveError(&UnmarshalTypeError{"number " + s, v.Type()})
				break
			}
			v.SetFloat(n)
		}
	}
}

// The xxxInterface routines build up a value to be stored
// in an empty interface.  They are not strictly necessary,
// but they avoid the weight of reflection in this common case.

// valueInterface is like value but returns interface{}
func (d *decodeState) valueInterface() interface{} {
	switch d.scanWhile(scanSkipSpace) {
	default:
		d.error(errPhase)
	case scanBeginArray:
		return d.arrayInterface()
	case scanBeginObject:
		return d.objectInterface()
	case scanBeginLiteral:
		return d.literalInterface()
	}
	panic("unreachable")
}

// arrayInterface is like array but returns []interface{}.
func (d *decodeState) arrayInterface() []interface{} {
	var v []interface{}
	for {
		// Look ahead for ] - can only happen on first iteration.
		op := d.scanWhile(scanSkipSpace)
		if op == scanEndArray {
			break
		}

		// Back up so d.value can have the byte we just read.
		d.off--
		d.scan.undo(op)

		v = append(v, d.valueInterface())

		// Next token must be , or ].
		op = d.scanWhile(scanSkipSpace)
		if op == scanEndArray {
			break
		}
		if op != scanArrayValue {
			d.error(errPhase)
		}
	}
	return v
}

// objectInterface is like object but returns map[string]interface{}.
func (d *decodeState) objectInterface() map[string]interface{} {
	m := make(map[string]interface{})
	for {
		// Read opening " of string key or closing }.
		op := d.scanWhile(scanSkipSpace)
		if op == scanEndObject {
			// closing } - can only happen on first iteration.
			break
		}
		if op != scanBeginLiteral {
			d.error(errPhase)
		}

		// Read string key.
		start := d.off - 1
		op = d.scanWhile(scanContinue)
		item := d.data[start : d.off-1]
		key, ok := unquote(item)
		if !ok {
			d.error(errPhase)
		}

		// Read : before value.
		if op == scanSkipSpace {
			op = d.scanWhile(scanSkipSpace)
		}
		if op != scanObjectKey {
			d.error(errPhase)
		}

		// Read value.
		m[key] = d.valueInterface()

		// Next token must be , or }.
		op = d.scanWhile(scanSkipSpace)
		if op == scanEndObject {
			break
		}
		if op != scanObjectValue {
			d.error(errPhase)
		}
	}
	return m
}

// literalInterface is like literal but returns an interface value.
func (d *decodeState) literalInterface() interface{} {
	// All bytes inside literal return scanContinue op code.
	start := d.off - 1
	op := d.scanWhile(scanContinue)

	// Scan read one byte too far; back up.
	d.off--
	d.scan.undo(op)
	item := d.data[start:d.off]

	switch c := item[0]; c {
	case 'n': // null
		return nil

	case 't', 'f': // true, false
		return c == 't'

	case '"': // string
		s, ok := unquote(item)
		if !ok {
			d.error(errPhase)
		}
		return s

	default: // number
		if c != '-' && (c < '0' || c > '9') {
			d.error(errPhase)
		}
		n, err := strconv.ParseFloat(string(item), 64)
		if err != nil {
			d.saveError(&UnmarshalTypeError{"number " + string(item), reflect.TypeOf(0.0)})
		}
		return n
	}
	panic("unreachable")
}

// getu4 decodes \uXXXX from the beginning of s, returning the hex value,
// or it returns -1.
func getu4(s []byte) rune {
	if len(s) < 6 || s[0] != '\\' || s[1] != 'u' {
		return -1
	}
	r, err := strconv.ParseUint(string(s[2:6]), 16, 64)
	if err != nil {
		return -1
	}
	return rune(r)
}

// unquote converts a quoted JSON string literal s into an actual string t.
// The rules are different than for Go, so cannot use strconv.Unquote.
func unquote(s []byte) (t string, ok bool) {
	s, ok = unquoteBytes(s)
	t = string(s)
	return
}

func unquoteBytes(s []byte) (t []byte, ok bool) {
	if len(s) < 2 || s[0] != '"' || s[len(s)-1] != '"' {
		return
	}
	s = s[1 : len(s)-1]

	// Check for unusual characters. If there are none,
	// then no unquoting is needed, so return a slice of the
	// original bytes.
	r := 0
	for r < len(s) {
		c := s[r]
		if c == '\\' || c == '"' || c < ' ' {
			break
		}
		if c < utf8.RuneSelf {
			r++
			continue
		}
		rr, size := utf8.DecodeRune(s[r:])
		if rr == utf8.RuneError && size == 1 {
			break
		}
		r += size
	}
	if r == len(s) {
		return s, true
	}

	b := make([]byte, len(s)+2*utf8.UTFMax)
	w := copy(b, s[0:r])
	for r < len(s) {
		// Out of room?  Can only happen if s is full of
		// malformed UTF-8 and we're replacing each
		// byte with RuneError.
		if w >= len(b)-2*utf8.UTFMax {
			nb := make([]byte, (len(b)+utf8.UTFMax)*2)
			copy(nb, b[0:w])
			b = nb
		}
		switch c := s[r]; {
		case c == '\\':
			r++
			if r >= len(s) {
				return
			}
			switch s[r] {
			default:
				return
			case '"', '\\', '/', '\'':
				b[w] = s[r]
				r++
				w++
			case 'b':
				b[w] = '\b'
				r++
				w++
			case 'f':
				b[w] = '\f'
				r++
				w++
			case 'n':
				b[w] = '\n'
				r++
				w++
			case 'r':
				b[w] = '\r'
				r++
				w++
			case 't':
				b[w] = '\t'
				r++
				w++
			case 'u':
				r--
				rr := getu4(s[r:])
				if rr < 0 {
					return
				}
				r += 6
				if utf16.IsSurrogate(rr) {
					rr1 := getu4(s[r:])
					if dec := utf16.DecodeRune(rr, rr1); dec != unicode.ReplacementChar {
						// A valid pair; consume.
						r += 6
						w += utf8.EncodeRune(b[w:], dec)
						break
					}
					// Invalid surrogate; fall back to replacement rune.
					rr = unicode.ReplacementChar
				}
				w += utf8.EncodeRune(b[w:], rr)
			}

		// Quote, control characters are invalid.
		case c == '"', c < ' ':
			return

		// ASCII
		case c < utf8.RuneSelf:
			b[w] = c
			r++
			w++

		// Coerce to well-formed UTF-8.
		default:
			rr, size := utf8.DecodeRune(s[r:])
			r += size
			w += utf8.EncodeRune(b[w:], rr)
		}
	}
	return b[0:w], true
}
