// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unify

import (
	"fmt"
	"reflect"
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
// Any type may implement [Decoder], in which case its DecodeUnified method will
// be called instead of using the default decoding scheme.
func (v *Value) Decode(into any) error {
	rv := reflect.ValueOf(into)
	if rv.Kind() != reflect.Pointer {
		return fmt.Errorf("cannot decode into non-pointer %T", into)
	}
	return decodeReflect(v, rv.Elem())
}

// Decoder can be implemented by types as a custom implementation of [Decode]
// for that type.
type Decoder interface {
	DecodeUnified(v *Value) error
}

var decoderType = reflect.TypeFor[Decoder]()

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
	if reflect.PointerTo(rv.Type()).Implements(decoderType) {
		// Use the custom decoder.
		err = rv.Addr().Interface().(Decoder).DecodeUnified(v)
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
		v := d.fields[defName]
		if v == nil {
			v = topValue
		}
		if err := decodeReflect(v, rv.FieldByIndex(f.Index)); err != nil {
			return newDecodeError(f.Name, err)
		}
	}
	return nil
}

var structFieldsCache sync.Map /*[reflect.Type, map[string]reflect.StructField]*/

// canonStructFields canonicalizes the name of all exported fields in rt to from
// Go-style exported names to YAML-style lower-case names. If a name starts with
// N upper-case letters, then if N==1, it lower-cases just the first letter; if
// N=len, it lower-cases the whole name; otherwise it lower-cases the first N-1
// letters.
//
// For example:
//
//	AsmPos      => asmPos
//	CPUFeatures => cpuFeatures
//	GOARCH      => goarch
//
// It returns a map from Def field name to struct field. The mapping between Go
// field names and Def names is a bijection, so it can be used for encoding and
// decoding.
//
// rt must be a struct type.
func canonStructFields(rt reflect.Type) map[string]reflect.StructField {
	type fieldMap = map[string]reflect.StructField
	if fields, ok := structFieldsCache.Load(rt); ok {
		return fields.(fieldMap)
	}

	fm := make(fieldMap)
	for f := range rt.Fields() {
		if !f.IsExported() {
			continue
		}
		defName := lowerGoName(f.Name)
		if _, ok := fm[defName]; ok {
			panic(fmt.Sprintf("multiple fields in type %s map to %q", rt, defName))
		}
		fm[defName] = f
	}

	res, _ := structFieldsCache.LoadOrStore(rt, fm)
	return res.(fieldMap)
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
