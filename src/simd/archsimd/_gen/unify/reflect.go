// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unify

import (
	"fmt"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"unicode"
)

// Decode decodes v into a Go value.
//
// v must be exact, except that it can include Top. into must be a pointer.
// [Def]s are decoded into structs. [Tuple]s are decoded into slices. [String]s
// are decoded into strings or ints. Any field can itself be a pointer to one of
// these types. Top can be decoded into a pointer-typed field and will set the
// field to nil. Anything else will allocate a value if necessary.
//
// Any type may implement [DecoderEncoder], in which case its DecodeUnified
// method will be called instead of using the default decoding scheme.
func (v *Value) Decode(into any) error {
	rv := reflect.ValueOf(into)
	if rv.Kind() != reflect.Pointer {
		return fmt.Errorf("cannot decode into non-pointer %T", into)
	}
	return decodeReflect(v, rv.Elem())
}

// Encode constructs a Value from a Go value. It is the inverse of
// [Value.Decode].
//
// If a struct has an "Encode<Field>" field, it will be used for encoding the
// field named by <Field>, overriding the default field. This behavior is useful
// when a single type is used as an input and an output from unification, but
// the input and output have different requirements (e.g., optionality or
// strings vs regexps).
func Encode(gv any) *Value {
	rv := reflect.ValueOf(gv)
	return encodeReflect(rv)
}

// DecoderEncoder can be implemented by types as a custom implementation of
// [Decode] and [Encode] for that type. These must be inverses.
type DecoderEncoder interface {
	DecodeUnified(v *Value) error
	EncodeUnified() *Value
}

var decoderEncoder = reflect.TypeFor[DecoderEncoder]()

func decodeReflect(v *Value, rv reflect.Value) error {
	var ptr reflect.Value
	if rv.Kind() == reflect.Pointer {
		if rv.IsNil() {
			// Transparently allocate through pointers, *except* for Top, which
			// wants to set the pointer to nil.
			//
			// TODO: Drop this condition if I switch to an explicit Optional[T]
			// or move the Top logic into Def.
			if _, ok := v.Domain.(Top); !ok {
				// Allocate the value to fill in, but don't actually store it in
				// the pointer until we successfully decode.
				ptr = rv
				rv = reflect.New(rv.Type().Elem()).Elem()
			}
		} else {
			rv = rv.Elem()
		}
	}

	var err error
	if reflect.PointerTo(rv.Type()).Implements(decoderEncoder) {
		// Use the custom decoder.
		err = rv.Addr().Interface().(DecoderEncoder).DecodeUnified(v)
	} else {
		err = v.Domain.decode(rv)
	}
	if err == nil && ptr.IsValid() {
		ptr.Set(rv.Addr())
	}
	return err
}

type inexactError struct {
	valueType string
	goType    string
}

func (e *inexactError) Error() string {
	return fmt.Sprintf("cannot store inexact %s value in %s", e.valueType, e.goType)
}

type decodeError struct {
	path string
	err  error
}

func newDecodeError(path string, err error) *decodeError {
	if err, ok := err.(*decodeError); ok {
		return &decodeError{path: path + "." + err.path, err: err.err}
	}
	return &decodeError{path: path, err: err}
}

func (e *decodeError) Unwrap() error {
	return e.err
}

func (e *decodeError) Error() string {
	return fmt.Sprintf("%s: %s", e.path, e.err)
}

func (d Var) decode(rv reflect.Value) error {
	return &inexactError{"var", rv.Type().String()}
}

func encodeReflect(rv reflect.Value) *Value {
	if rv.Kind() == reflect.Pointer {
		// Transparently read through non-nil pointers
		if rv.IsNil() {
			return topValue
		}
		if re, ok := rv.Interface().(*regexp.Regexp); ok {
			if exact, complete := re.LiteralPrefix(); complete {
				return NewValue(NewStringExact(exact))
			}
			return NewValue(String{kind: stringRegex, re: []*regexp.Regexp{re}})
		}
		rv = rv.Elem()
	}

	if reflect.PointerTo(rv.Type()).Implements(decoderEncoder) {
		// Use the custom encoder.
		return rv.Addr().Interface().(DecoderEncoder).EncodeUnified()
	}

	switch rv.Kind() {
	default:
		panic(fmt.Sprintf("cannot encode type %s to a unify.Value", rv.Type()))

	case reflect.Struct:
		var db DefBuilder
		fieldMap := canonStructFields(rv.Type())
		for defName, f := range fieldMap {
			if f.encode.Index == nil {
				continue
			}
			fVal := rv.FieldByIndex(f.encode.Index)
			if fVal.Kind() == reflect.Pointer && fVal.IsNil() {
				// Omit nil pointers from def (equivalent to setting them to
				// "top")
				continue
			}
			uv := encodeReflect(fVal)
			db.Add(defName, uv)
		}
		return NewValue(db.Build())

	case reflect.Slice:
		uvs := make([]*Value, rv.Len())
		for i := range rv.Len() {
			uvs[i] = encodeReflect(rv.Index(i))
		}
		return NewValue(NewTuple(uvs...))

	case reflect.String, reflect.Int, reflect.Bool:
		return NewValue(NewStringExact(fmt.Sprint(rv)))
	}
}

func (t Top) decode(rv reflect.Value) error {
	// We can decode Top into a pointer-typed value as nil.
	if rv.Kind() != reflect.Pointer {
		return &inexactError{"top", rv.Type().String()}
	}
	rv.SetZero()
	return nil
}

func (d Def) decode(rv reflect.Value) error {
	if rv.Kind() != reflect.Struct {
		return fmt.Errorf("cannot decode Def into %s", rv.Type())
	}

	fieldMap := canonStructFields(rv.Type())
	for defName, f := range fieldMap {
		if f.decode.Index == nil {
			continue
		}
		v := d.fields[defName]
		if v == nil {
			v = topValue
		}
		if err := decodeReflect(v, rv.FieldByIndex(f.decode.Index)); err != nil {
			return newDecodeError(f.decode.Name, err)
		}
	}
	return nil
}

type structFieldPair struct {
	decode reflect.StructField
	encode reflect.StructField
}

var structFieldsCache sync.Map /*[reflect.Type, map[string]structFieldPair]*/

// canonStructFields canonicalizes the name of all exported fields in rt from
// Go-style exported names to YAML-style lower-case names. If the Go name starts
// with N upper-case letters, then if N==1, it lower-cases just the first
// letter; if N=len, it lower-cases the whole name; otherwise it lower-cases the
// first N-1 letters.
//
// It produces two maps: one for encoding and one for decoding. By default,
// these are the same, but if a field has form "EncodeX", then it's used for
// encoding x.
//
// It returns a map from Def field name to a pair of decode/encode struct
// fields. The mapping between Go field names and Def names is a bijection, so
// it can be used for encoding and decoding.
//
// For example:
//
//	YAML field        Go field
//	asmPos        <=> AsmPos
//	cpuFeatures   <=> CPUFeatures
//	goarch        <=> GOARCH
//	bits          <=> Bits (decode) & EncodeBits (encode)
//
// rt must be a struct type.
func canonStructFields(rt reflect.Type) map[string]structFieldPair {
	type fieldMap = map[string]structFieldPair
	if fields, ok := structFieldsCache.Load(rt); ok {
		return fields.(fieldMap)
	}

	fm := make(fieldMap)
	for f := range rt.Fields() {
		if !f.IsExported() {
			continue
		}
		if isEncodeOverride(f.Name) {
			defName := lowerGoName(f.Name[6:])
			pair := fm[defName]
			pair.encode = f
			fm[defName] = pair
		} else {
			defName := lowerGoName(f.Name)
			pair := fm[defName]
			if pair.decode.Index != nil {
				panic(fmt.Sprintf("multiple fields in type %s map to %q", rt, defName))
			}
			pair.decode = f
			if pair.encode.Index == nil {
				pair.encode = f
			}
			fm[defName] = pair
		}
	}

	res, _ := structFieldsCache.LoadOrStore(rt, fm)
	return res.(fieldMap)
}

func isEncodeOverride(name string) bool {
	return strings.HasPrefix(name, "Encode") && len(name) > 6 && unicode.IsUpper(rune(name[6]))
}

func lowerGoName(goName string) string {
	prefixBytes := -1
	prevBytes := 0
	allUpper := true
	for pos, ch := range goName {
		if !unicode.IsUpper(ch) {
			allUpper = false
			prefixBytes = pos
			break
		}
		prevBytes = pos
	}
	if allUpper {
		// The whole name is upper-case.
		return strings.ToLower(goName)
	}
	if prevBytes == 0 {
		// The name starts with a single upper-case letter. Lower-case just it.
		prevBytes = prefixBytes
	}
	// Lower case the first n-1 upper-case letters.
	return strings.ToLower(goName[:prevBytes]) + goName[prevBytes:]
}

func (d Tuple) decode(rv reflect.Value) error {
	if d.repeat != nil {
		return &inexactError{"repeated tuple", rv.Type().String()}
	}
	// TODO: We could also do arrays.
	if rv.Kind() != reflect.Slice {
		return fmt.Errorf("cannot decode Tuple into %s", rv.Type())
	}
	if rv.IsNil() || rv.Cap() < len(d.vs) {
		rv.Set(reflect.MakeSlice(rv.Type(), len(d.vs), len(d.vs)))
	} else {
		rv.SetLen(len(d.vs))
	}
	for i, v := range d.vs {
		if err := decodeReflect(v, rv.Index(i)); err != nil {
			return newDecodeError(fmt.Sprintf("%d", i), err)
		}
	}
	return nil
}

func (d String) decode(rv reflect.Value) error {
	if d.kind != stringExact {
		return &inexactError{"regex", rv.Type().String()}
	}
	switch rv.Kind() {
	default:
		return fmt.Errorf("cannot decode String into %s", rv.Type())
	case reflect.String:
		rv.SetString(d.exact)
	case reflect.Int:
		i, err := strconv.Atoi(d.exact)
		if err != nil {
			return fmt.Errorf("cannot decode String into %s: %s", rv.Type(), err)
		}
		rv.SetInt(int64(i))
	case reflect.Bool:
		b, err := strconv.ParseBool(d.exact)
		if err != nil {
			return fmt.Errorf("cannot decode String into %s: %s", rv.Type(), err)
		}
		rv.SetBool(b)
	}
	return nil
}
