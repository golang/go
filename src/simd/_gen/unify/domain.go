// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unify

import (
	"fmt"
	"iter"
	"maps"
	"reflect"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

// A Domain is a non-empty set of values, all of the same kind.
//
// Domain may be a scalar:
//
//   - [String] - Represents string-typed values.
//
// Or a composite:
//
//   - [Def] - A mapping from fixed keys to [Domain]s.
//
//   - [Tuple] - A fixed-length sequence of [Domain]s or
//     all possible lengths repeating a [Domain].
//
// Or top or bottom:
//
//   - [Top] - Represents all possible values of all kinds.
//
//   - nil - Represents no values.
//
// Or a variable:
//
//   - [Var] - A value captured in the environment.
type Domain interface {
	Exact() bool
	WhyNotExact() string

	// decode stores this value in a Go value. If this value is not exact, this
	// returns a potentially wrapped *inexactError.
	decode(reflect.Value) error
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

// Top represents all possible values of all possible types.
type Top struct{}

func (t Top) Exact() bool         { return false }
func (t Top) WhyNotExact() string { return "is top" }

func (t Top) decode(rv reflect.Value) error {
	// We can decode Top into a pointer-typed value as nil.
	if rv.Kind() != reflect.Pointer {
		return &inexactError{"top", rv.Type().String()}
	}
	rv.SetZero()
	return nil
}

// A Def is a mapping from field names to [Value]s. Any fields not explicitly
// listed have [Value] [Top].
type Def struct {
	fields map[string]*Value
}

// A DefBuilder builds a [Def] one field at a time. The zero value is an empty
// [Def].
type DefBuilder struct {
	fields map[string]*Value
}

func (b *DefBuilder) Add(name string, v *Value) {
	if b.fields == nil {
		b.fields = make(map[string]*Value)
	}
	if _, ok := b.fields[name]; ok {
		panic(fmt.Sprintf("duplicate field %q", name))
	}
	b.fields[name] = v
}

// Build constructs a [Def] from the fields added to this builder.
func (b *DefBuilder) Build() Def {
	return Def{maps.Clone(b.fields)}
}

// Exact returns true if all field Values are exact.
func (d Def) Exact() bool {
	for _, v := range d.fields {
		if !v.Exact() {
			return false
		}
	}
	return true
}

// WhyNotExact returns why the value is not exact
func (d Def) WhyNotExact() string {
	for s, v := range d.fields {
		if !v.Exact() {
			w := v.WhyNotExact()
			return "field " + s + ": " + w
		}
	}
	return ""
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

func (d Def) keys() []string {
	return slices.Sorted(maps.Keys(d.fields))
}

func (d Def) All() iter.Seq2[string, *Value] {
	// TODO: We call All fairly often. It's probably bad to sort this every
	// time.
	keys := slices.Sorted(maps.Keys(d.fields))
	return func(yield func(string, *Value) bool) {
		for _, k := range keys {
			if !yield(k, d.fields[k]) {
				return
			}
		}
	}
}

// A Tuple is a sequence of Values in one of two forms: 1. a fixed-length tuple,
// where each Value can be different or 2. a "repeated tuple", which is a Value
// repeated 0 or more times.
type Tuple struct {
	vs []*Value

	// repeat, if non-nil, means this Tuple consists of an element repeated 0 or
	// more times. If repeat is non-nil, vs must be nil. This is a generator
	// function because we don't necessarily want *exactly* the same Value
	// repeated. For example, in YAML encoding, a !sum in a repeated tuple needs
	// a fresh variable in each instance.
	repeat []func(envSet) (*Value, envSet)
}

func NewTuple(vs ...*Value) Tuple {
	return Tuple{vs: vs}
}

func NewRepeat(gens ...func(envSet) (*Value, envSet)) Tuple {
	return Tuple{repeat: gens}
}

func (d Tuple) Exact() bool {
	if d.repeat != nil {
		return false
	}
	for _, v := range d.vs {
		if !v.Exact() {
			return false
		}
	}
	return true
}

func (d Tuple) WhyNotExact() string {
	if d.repeat != nil {
		return "d.repeat is not nil"
	}
	for i, v := range d.vs {
		if !v.Exact() {
			w := v.WhyNotExact()
			return "index " + strconv.FormatInt(int64(i), 10) + ": " + w
		}
	}
	return ""
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

// A String represents a set of strings. It can represent the intersection of a
// set of regexps, or a single exact string. In general, the domain of a String
// is non-empty, but we do not attempt to prove emptiness of a regexp value.
type String struct {
	kind  stringKind
	re    []*regexp.Regexp // Intersection of regexps
	exact string
}

type stringKind int

const (
	stringRegex stringKind = iota
	stringExact
)

func NewStringRegex(exprs ...string) (String, error) {
	if len(exprs) == 0 {
		exprs = []string{""}
	}
	v := String{kind: -1}
	for _, expr := range exprs {
		if expr == "" {
			// Skip constructing the regexp. It won't have a "literal prefix"
			// and so we wind up thinking this is a regexp instead of an exact
			// (empty) string.
			v = String{kind: stringExact, exact: ""}
			continue
		}

		re, err := regexp.Compile(`\A(?:` + expr + `)\z`)
		if err != nil {
			return String{}, fmt.Errorf("parsing value: %s", err)
		}

		// An exact value narrows the whole domain to exact, so we're done, but
		// should keep parsing.
		if v.kind == stringExact {
			continue
		}

		if exact, complete := re.LiteralPrefix(); complete {
			v = String{kind: stringExact, exact: exact}
		} else {
			v.kind = stringRegex
			v.re = append(v.re, re)
		}
	}
	return v, nil
}

func NewStringExact(s string) String {
	return String{kind: stringExact, exact: s}
}

// Exact returns whether this Value is known to consist of a single string.
func (d String) Exact() bool {
	return d.kind == stringExact
}

func (d String) WhyNotExact() string {
	if d.kind == stringExact {
		return ""
	}
	return "string is not exact"
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
