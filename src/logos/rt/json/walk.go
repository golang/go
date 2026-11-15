// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"encoding"
	"encoding/base64"
	stdjson "encoding/json"
	"fmt"
	"reflect"
	"slices"
	"strconv"
	"strings"
)

// Walk is one root MarshalJSON call's traversal state. The zero value
// is ready: Gen is filled by generated code before any stateful edge,
// and the seen set is created lazily past the threshold so the flat
// scalar path never allocates.
type Walk struct {
	// Gen holds the exact generated struct types this root may meet
	// through interface or reflective edges, as built by generated
	// AddType calls. Membership decides dispatch, never output.
	Gen      []reflect.Type
	fields   map[reflect.Type][]walkField
	ptrLevel uint
	depth    uint
	seen     map[any]struct{}
	depthErr *stdjson.SyntaxError
}

type walkVisit struct {
	typ reflect.Type
	ptr any
	len int
}

// AddType records t and its pointer/value counterpart for dispatch. A
// nil t is ignored so generated code needs no nil guard around
// TypeOf on interface values. Generated code emits AddType only for
// statically typed paths the compiler knows are generated-side (plus
// the root pair); interface-typed paths never contribute, so a user
// method type met only dynamically can never enter Gen.
func (s *Walk) AddType(t reflect.Type) {
	if t == nil {
		return
	}
	s.Gen = append(s.Gen, t)
	if t.Kind() == reflect.Pointer {
		s.Gen = append(s.Gen, t.Elem())
	} else {
		s.Gen = append(s.Gen, reflect.PointerTo(t))
	}
}

func (s *Walk) inGen(t reflect.Type) bool {
	for _, g := range s.Gen {
		if g == t {
			return true
		}
	}
	return false
}

func cycleError(v reflect.Value) error {
	return &stdjson.UnsupportedValueError{Value: v, Str: "encountered a cycle via " + v.Type().String()}
}

// EnterFrame records one object or array frame. offset is relative to
// the current encoding buffer; helper callers rebase a depth error as
// it unwinds through their exact emitted prefixes.
func (s *Walk) EnterFrame(offset int64) error {
	if err := s.checkDepth(offset); err != nil {
		return err
	}
	s.depth++
	return nil
}

// RebaseDepthError translates this walk's own depth error from a nested
// value buffer to its caller's buffer. User errors, including user
// SyntaxErrors, are returned untouched.
func (s *Walk) RebaseDepthError(err error, prefix int64) error {
	if s.depthErr != nil && err == s.depthErr {
		s.depthErr.Offset += prefix
	}
	return err
}

// LeaveFrame closes an enterFrame edge.
func (s *Walk) LeaveFrame() {
	s.depth--
}

// Enter records a pointer, map, or slice edge for the value held in p,
// reporting the reflective cycle error on a repeat visit past the
// threshold. Generated code calls Enter with the edge value itself;
// other kinds are ignored so sites need no kind guard.
func (s *Walk) Enter(p any) error {
	v := reflect.ValueOf(p)
	switch v.Kind() {
	case reflect.Pointer, reflect.Map, reflect.Slice:
	default:
		return nil
	}
	return s.enterValue(v)
}

// Leave closes an Enter edge, dropping the visit from the ancestor set.
func (s *Walk) Leave(p any) {
	v := reflect.ValueOf(p)
	switch v.Kind() {
	case reflect.Pointer, reflect.Map, reflect.Slice:
	default:
		return
	}
	s.leaveValue(v)
}

func (s *Walk) enterValue(v reflect.Value) error {
	s.ptrLevel++
	if s.ptrLevel > 1000 {
		k := visitKey(v)
		if _, ok := s.seen[k]; ok {
			s.ptrLevel--
			return cycleError(v)
		}
		if s.seen == nil {
			s.seen = make(map[any]struct{})
		}
		s.seen[k] = struct{}{}
	}
	return nil
}

func (s *Walk) leaveValue(v reflect.Value) {
	if s.ptrLevel > 1000 {
		delete(s.seen, visitKey(v))
	}
	s.ptrLevel--
}

// Marshal encodes v with shared traversal state, returning fresh bytes
// like MarshalField. It is the funnel for every stateful edge the
// generated helpers cannot spell: slices, arrays, maps, interfaces, and
// unspellable or dynamic struct values.
func (s *Walk) Marshal(v any) ([]byte, error) {
	buf, err := s.appendValue(nil, reflect.ValueOf(v))
	if err != nil {
		return nil, err
	}
	return buf, nil
}

func (s *Walk) appendValue(buf []byte, rv reflect.Value) ([]byte, error) {
	if !rv.IsValid() {
		return append(buf, "null"...), nil
	}
	t := rv.Type()
	if t == numberType {
		return appendNumber(buf, rv)
	}
	if s.honorsJSONMethod(rv) {
		return appendHonored(buf, rv)
	}
	switch rv.Kind() {
	case reflect.Interface:
		if rv.IsNil() {
			return append(buf, "null"...), nil
		}
		return s.appendValue(buf, rv.Elem())
	case reflect.Pointer:
		if rv.IsNil() {
			return append(buf, "null"...), nil
		}
		if err := s.enterValue(rv); err != nil {
			return buf, err
		}
		buf, err := s.appendValue(buf, rv.Elem())
		s.leaveValue(rv)
		return buf, err
	case reflect.Bool:
		if rv.Bool() {
			return append(buf, "true"...), nil
		}
		return append(buf, "false"...), nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return AppendInt(buf, rv.Int()), nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return AppendUint(buf, rv.Uint()), nil
	case reflect.Float32, reflect.Float64:
		bits := int(t.Size()) * 8
		f := rv.Float()
		if f-f != 0 {
			return buf, FloatErr(rv.Interface(), f, bits)
		}
		return AppendFloat(buf, f, bits), nil
	case reflect.String:
		buf = append(buf, '"')
		buf = AppendEscaped(buf, rv.String())
		return append(buf, '"'), nil
	case reflect.Slice:
		if rv.IsNil() {
			return append(buf, "null"...), nil
		}
		if rv.Type().Elem().Kind() == reflect.Uint8 && !sliceElemHasJSONMethod(rv.Type().Elem()) {
			buf = append(buf, '"')
			buf = append(buf, base64.StdEncoding.EncodeToString(rv.Bytes())...)
			return append(buf, '"'), nil
		}
		if err := s.EnterFrame(int64(len(buf))); err != nil {
			return buf, err
		}
		defer s.LeaveFrame()
		if err := s.enterValue(rv); err != nil {
			return buf, err
		}
		buf = append(buf, '[')
		for i := range rv.Len() {
			if i > 0 {
				buf = append(buf, ',')
			}
			var err error
			buf, err = s.appendValue(buf, rv.Index(i))
			if err != nil {
				s.leaveValue(rv)
				return buf, err
			}
		}
		buf = append(buf, ']')
		s.leaveValue(rv)
		return buf, nil
	case reflect.Array:
		if err := s.EnterFrame(int64(len(buf))); err != nil {
			return buf, err
		}
		defer s.LeaveFrame()
		buf = append(buf, '[')
		for i := range rv.Len() {
			if i > 0 {
				buf = append(buf, ',')
			}
			var err error
			buf, err = s.appendValue(buf, rv.Index(i))
			if err != nil {
				return buf, err
			}
		}
		return append(buf, ']'), nil
	case reflect.Map:
		if err := s.checkMapKeys(rv); err != nil {
			return buf, err
		}
		if rv.IsNil() {
			return append(buf, "null"...), nil
		}
		if err := s.EnterFrame(int64(len(buf))); err != nil {
			return buf, err
		}
		defer s.LeaveFrame()
		if err := s.enterValue(rv); err != nil {
			return buf, err
		}
		buf, err := s.appendMap(buf, rv)
		s.leaveValue(rv)
		return buf, err
	case reflect.Struct:
		return s.appendStruct(buf, rv)
	}
	return buf, &stdjson.UnsupportedTypeError{Type: t}
}

var numberType = reflect.TypeFor[stdjson.Number]()

func numberBody(rv reflect.Value) ([]byte, error) {
	return MarshalField(stdjson.Number(rv.String()))
}

func appendNumber(buf []byte, rv reflect.Value) ([]byte, error) {
	b, err := numberBody(rv)
	if err != nil {
		return buf, err
	}
	return append(buf, b...), nil
}

func appendQuotedNumber(buf []byte, rv reflect.Value) ([]byte, error) {
	b, err := numberBody(rv)
	if err != nil {
		return buf, err
	}
	buf = append(buf, '"')
	buf = append(buf, b...)
	return append(buf, '"'), nil
}

func (s *Walk) appendStringOptionValue(buf []byte, rv reflect.Value) ([]byte, error) {
	if rv.Type() == numberType {
		return appendQuotedNumber(buf, rv)
	}
	if rv.Kind() == reflect.Pointer && !rv.IsNil() && rv.Type().Elem() == numberType {
		return appendQuotedNumber(buf, rv.Elem())
	}
	if s.honorsJSONMethod(rv) {
		return appendHonored(buf, rv)
	}
	if rv.Kind() == reflect.Pointer {
		if rv.IsNil() {
			return append(buf, "null"...), nil
		}
		return s.appendStringOptionValue(buf, rv.Elem())
	}
	switch rv.Kind() {
	case reflect.Bool:
		if rv.Bool() {
			return append(buf, `"true"`...), nil
		}
		return append(buf, `"false"`...), nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		buf = append(buf, '"')
		buf = AppendInt(buf, rv.Int())
		return append(buf, '"'), nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		buf = append(buf, '"')
		buf = AppendUint(buf, rv.Uint())
		return append(buf, '"'), nil
	case reflect.Float32, reflect.Float64:
		bits := int(rv.Type().Size()) * 8
		f := rv.Float()
		if f-f != 0 {
			return buf, FloatErr(rv.Interface(), f, bits)
		}
		buf = append(buf, '"')
		buf = AppendFloat(buf, f, bits)
		return append(buf, '"'), nil
	case reflect.String:
		buf = append(buf, '"')
		buf = AppendQuoted(buf, rv.String())
		return append(buf, '"'), nil
	}
	return s.appendValue(buf, rv)
}

func (s *Walk) honorsJSONMethod(rv reflect.Value) bool {
	return !s.inGen(rv.Type()) && hasJSONMethod(rv)
}

func appendHonored(buf []byte, rv reflect.Value) ([]byte, error) {
	b, err := MarshalField(addrWalkInterface(rv))
	if err != nil {
		return buf, err
	}
	return append(buf, b...), nil
}

func addrWalkInterface(rv reflect.Value) any {
	t := rv.Type()
	for _, iface := range walkMethodTypes {
		if t.Implements(iface) {
			return rv.Interface()
		}
		if t.Kind() != reflect.Pointer && rv.CanAddr() && reflect.PointerTo(t).Implements(iface) {
			return rv.Addr().Interface()
		}
	}
	return rv.Interface()
}

func hasJSONMethod(rv reflect.Value) bool {
	t := rv.Type()
	for _, iface := range walkMethodTypes {
		if t.Implements(iface) {
			return true
		}
		if t.Kind() != reflect.Pointer && rv.CanAddr() && reflect.PointerTo(t).Implements(iface) {
			return true
		}
	}
	return false
}

func sliceElemHasJSONMethod(elem reflect.Type) bool {
	for _, iface := range walkMethodTypes {
		if elem.Implements(iface) {
			return true
		}
		if reflect.PointerTo(elem).Implements(iface) {
			return true
		}
	}
	return false
}

var (
	marshalerType     = reflect.TypeFor[stdjson.Marshaler]()
	textAppenderType  = reflect.TypeFor[encoding.TextAppender]()
	textMarshalerType = reflect.TypeFor[encoding.TextMarshaler]()
)

func walkMapKeySupported(t reflect.Type) bool {
	switch t.Kind() {
	case reflect.String,
		reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return true
	}
	return t.Implements(textMarshalerType)
}

func (s *Walk) appendMap(buf []byte, rv reflect.Value) ([]byte, error) {
	type kv struct {
		ks string
		v  reflect.Value
	}
	kvs := make([]kv, 0, rv.Len())
	mi := rv.MapRange()
	for mi.Next() {
		ks, err := resolveWalkKey(mi.Key())
		if err != nil {
			return buf, fmt.Errorf("json: encoding error for type %q: %q", rv.Type().String(), err.Error())
		}
		kvs = append(kvs, kv{ks: ks, v: mi.Value()})
	}
	slices.SortFunc(kvs, func(a, b kv) int { return strings.Compare(a.ks, b.ks) })
	buf = append(buf, '{')
	for i := range kvs {
		if i > 0 {
			buf = append(buf, ',')
		}
		buf = append(buf, '"')
		buf = AppendEscaped(buf, kvs[i].ks)
		buf = append(buf, '"', ':')
		var err error
		buf, err = s.appendValue(buf, kvs[i].v)
		if err != nil {
			return buf, err
		}
	}
	return append(buf, '}'), nil
}

func resolveWalkKey(k reflect.Value) (string, error) {
	if k.Kind() == reflect.String {
		return k.String(), nil
	}
	if tm, ok := reflect.TypeAssert[encoding.TextMarshaler](k); ok {
		if k.Kind() == reflect.Pointer && k.IsNil() {
			return "", nil
		}
		text, err := tm.MarshalText()
		if err != nil {
			return "", err
		}
		return string(text), nil
	}
	switch k.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return strconv.FormatInt(k.Int(), 10), nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return strconv.FormatUint(k.Uint(), 10), nil
	}
	panic("unexpected map key type")
}
