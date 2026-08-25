// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unify

import (
	"fmt"
	"reflect"
	"strconv"
	"strings"
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

	var lowered map[string]string // Lower case -> canonical for d.fields.
	rt := rv.Type()
	for fi := range rv.NumField() {
		fType := rt.Field(fi)
		if fType.PkgPath != "" {
			continue
		}
		v := d.fields[fType.Name]
		if v == nil {
			v = topValue

			// Try a case-insensitive match
			canon, ok := d.fields[strings.ToLower(fType.Name)]
			if ok {
				v = canon
			} else {
				if lowered == nil {
					lowered = make(map[string]string, len(d.fields))
					for k := range d.fields {
						l := strings.ToLower(k)
						if k != l {
							lowered[l] = k
						}
					}
				}
				canon, ok := lowered[strings.ToLower(fType.Name)]
				if ok {
					v = d.fields[canon]
				}
			}
		}
		if err := decodeReflect(v, rv.Field(fi)); err != nil {
			return newDecodeError(fType.Name, err)
		}
	}
	return nil
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
