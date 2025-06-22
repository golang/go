// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package json

import (
	"bytes"
	"errors"
	"io"
	"reflect"

	"encoding/json/internal/jsonflags"
	"encoding/json/internal/jsonopts"
	"encoding/json/internal/jsonwire"
	"encoding/json/jsontext"
)

// This package supports "inlining" a Go struct field, where the contents
// of the serialized field (which must be a JSON object) are treated as if
// they are part of the parent Go struct (which represents a JSON object).
//
// Generally, inlined fields are of a Go struct type, where the fields of the
// nested struct are virtually hoisted up to the parent struct using rules
// similar to how Go embedding works (but operating within the JSON namespace).
//
// However, inlined fields may also be of a Go map type with a string key or
// a jsontext.Value. Such inlined fields are called "fallback" fields since they
// represent any arbitrary JSON object member. Explicitly named fields take
// precedence over the inlined fallback. Only one inlined fallback is allowed.

var errRawInlinedNotObject = errors.New("inlined raw value must be a JSON object")

var jsontextValueType = reflect.TypeFor[jsontext.Value]()

// marshalInlinedFallbackAll marshals all the members in an inlined fallback.
func marshalInlinedFallbackAll(enc *jsontext.Encoder, va addressableValue, mo *jsonopts.Struct, f *structField, insertUnquotedName func([]byte) bool) error {
	v := addressableValue{va.Field(f.index0), va.forcedAddr} // addressable if struct value is addressable
	if len(f.index) > 0 {
		v = v.fieldByIndex(f.index, false)
		if !v.IsValid() {
			return nil // implies a nil inlined field
		}
	}
	v = v.indirect(false)
	if !v.IsValid() {
		return nil
	}

	if v.Type() == jsontextValueType {
		b, _ := reflect.TypeAssert[jsontext.Value](v.Value)
		if len(b) == 0 { // TODO: Should this be nil? What if it were all whitespace?
			return nil
		}

		dec := export.GetBufferedDecoder(b)
		defer export.PutBufferedDecoder(dec)
		xd := export.Decoder(dec)
		xd.Flags.Set(jsonflags.AllowDuplicateNames | jsonflags.AllowInvalidUTF8 | 1)

		tok, err := dec.ReadToken()
		if err != nil {
			if err == io.EOF {
				err = io.ErrUnexpectedEOF
			}
			return newMarshalErrorBefore(enc, v.Type(), err)
		}
		if tok.Kind() != '{' {
			return newMarshalErrorBefore(enc, v.Type(), errRawInlinedNotObject)
		}
		for dec.PeekKind() != '}' {
			// Parse the JSON object name.
			var flags jsonwire.ValueFlags
			val, err := xd.ReadValue(&flags)
			if err != nil {
				return newMarshalErrorBefore(enc, v.Type(), err)
			}
			if insertUnquotedName != nil {
				name := jsonwire.UnquoteMayCopy(val, flags.IsVerbatim())
				if !insertUnquotedName(name) {
					return newDuplicateNameError(enc.StackPointer().Parent(), val, enc.OutputOffset())
				}
			}
			if err := enc.WriteValue(val); err != nil {
				return err
			}

			// Parse the JSON object value.
			val, err = xd.ReadValue(&flags)
			if err != nil {
				return newMarshalErrorBefore(enc, v.Type(), err)
			}
			if err := enc.WriteValue(val); err != nil {
				return err
			}
		}
		if _, err := dec.ReadToken(); err != nil {
			return newMarshalErrorBefore(enc, v.Type(), err)
		}
		if err := xd.CheckEOF(); err != nil {
			return newMarshalErrorBefore(enc, v.Type(), err)
		}
		return nil
	} else {
		m := v // must be a map[~string]V
		n := m.Len()
		if n == 0 {
			return nil
		}
		mk := newAddressableValue(m.Type().Key())
		mv := newAddressableValue(m.Type().Elem())
		marshalKey := func(mk addressableValue) error {
			b, err := jsonwire.AppendQuote(enc.AvailableBuffer(), mk.String(), &mo.Flags)
			if err != nil {
				return newMarshalErrorBefore(enc, m.Type().Key(), err)
			}
			if insertUnquotedName != nil {
				isVerbatim := bytes.IndexByte(b, '\\') < 0
				name := jsonwire.UnquoteMayCopy(b, isVerbatim)
				if !insertUnquotedName(name) {
					return newDuplicateNameError(enc.StackPointer().Parent(), b, enc.OutputOffset())
				}
			}
			return enc.WriteValue(b)
		}
		marshalVal := f.fncs.marshal
		if mo.Marshalers != nil {
			marshalVal, _ = mo.Marshalers.(*Marshalers).lookup(marshalVal, mv.Type())
		}
		if !mo.Flags.Get(jsonflags.Deterministic) || n <= 1 {
			for iter := m.MapRange(); iter.Next(); {
				mk.SetIterKey(iter)
				if err := marshalKey(mk); err != nil {
					return err
				}
				mv.Set(iter.Value())
				if err := marshalVal(enc, mv, mo); err != nil {
					return err
				}
			}
		} else {
			names := getStrings(n)
			for i, iter := 0, m.Value.MapRange(); i < n && iter.Next(); i++ {
				mk.SetIterKey(iter)
				(*names)[i] = mk.String()
			}
			names.Sort()
			for _, name := range *names {
				mk.SetString(name)
				if err := marshalKey(mk); err != nil {
					return err
				}
				// TODO(https://go.dev/issue/57061): Use mv.SetMapIndexOf.
				mv.Set(m.MapIndex(mk.Value))
				if err := marshalVal(enc, mv, mo); err != nil {
					return err
				}
			}
			putStrings(names)
		}
		return nil
	}
}

// unmarshalInlinedFallbackNext unmarshals only the next member in an inlined fallback.
func unmarshalInlinedFallbackNext(dec *jsontext.Decoder, va addressableValue, uo *jsonopts.Struct, f *structField, quotedName, unquotedName []byte) error {
	v := addressableValue{va.Field(f.index0), va.forcedAddr} // addressable if struct value is addressable
	if len(f.index) > 0 {
		v = v.fieldByIndex(f.index, true)
	}
	v = v.indirect(true)

	if v.Type() == jsontextValueType {
		b, _ := reflect.TypeAssert[*jsontext.Value](v.Addr())
		if len(*b) == 0 { // TODO: Should this be nil? What if it were all whitespace?
			*b = append(*b, '{')
		} else {
			*b = jsonwire.TrimSuffixWhitespace(*b)
			if jsonwire.HasSuffixByte(*b, '}') {
				// TODO: When merging into an object for the first time,
				// should we verify that it is valid?
				*b = jsonwire.TrimSuffixByte(*b, '}')
				*b = jsonwire.TrimSuffixWhitespace(*b)
				if !jsonwire.HasSuffixByte(*b, ',') && !jsonwire.HasSuffixByte(*b, '{') {
					*b = append(*b, ',')
				}
			} else {
				return newUnmarshalErrorAfterWithSkipping(dec, v.Type(), errRawInlinedNotObject)
			}
		}
		*b = append(*b, quotedName...)
		*b = append(*b, ':')
		val, err := dec.ReadValue()
		if err != nil {
			return err
		}
		*b = append(*b, val...)
		*b = append(*b, '}')
		return nil
	} else {
		name := string(unquotedName) // TODO: Intern this?

		m := v // must be a map[~string]V
		if m.IsNil() {
			m.Set(reflect.MakeMap(m.Type()))
		}
		mk := reflect.ValueOf(name)
		if mkt := m.Type().Key(); mkt != stringType {
			mk = mk.Convert(mkt)
		}
		mv := newAddressableValue(m.Type().Elem()) // TODO: Cache across calls?
		if v2 := m.MapIndex(mk); v2.IsValid() {
			mv.Set(v2)
		}

		unmarshal := f.fncs.unmarshal
		if uo.Unmarshalers != nil {
			unmarshal, _ = uo.Unmarshalers.(*Unmarshalers).lookup(unmarshal, mv.Type())
		}
		err := unmarshal(dec, mv, uo)
		m.SetMapIndex(mk, mv.Value)
		if err != nil {
			return err
		}
		return nil
	}
}
