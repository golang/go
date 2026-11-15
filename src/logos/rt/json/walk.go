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
	"unsafe"
)

// Walk is one root MarshalJSON call's traversal state. The zero value
// is ready, and the seen set is created lazily past the threshold so the
// flat scalar path never allocates.
type Walk struct {
	// Gen encodes a dynamic value whose type is one of the generated
	// struct types the root may meet, or a pointer to one, appending to
	// the buffer with this walk's state. It reports false for any other
	// type. Generated code sets it before any stateful edge.
	Gen      func(*Walk, []byte, any) ([]byte, bool, error)
	fields   map[reflect.Type][]walkField
	ptrLevel uint
	depth    uint
	seen     map[walkVisit]struct{}
	depthErr *stdjson.SyntaxError
}

type walkVisit struct {
	typ any
	ptr unsafe.Pointer
	len int
}

const startDetectingCyclesAfter = 1000

func typeToken[V any]() any {
	return (*V)(nil)
}

func valueToken(t reflect.Type) any {
	return reflect.Zero(reflect.PointerTo(t)).Interface()
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

// EnterElems records the frame of a slice or map holding n elements.
// encoding/json writes an empty one as a single token without opening
// a frame, so only a non-empty one counts against the depth bound.
func (s *Walk) EnterElems(offset int64, n int) error {
	if n == 0 {
		s.depth++
		return nil
	}
	return s.EnterFrame(offset)
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

// LeaveFrame closes an EnterFrame edge.
func (s *Walk) LeaveFrame() {
	s.depth--
}

func (s *Walk) visit(key walkVisit) bool {
	if _, ok := s.seen[key]; ok {
		s.ptrLevel--
		return false
	}
	if s.seen == nil {
		s.seen = make(map[walkVisit]struct{})
	}
	s.seen[key] = struct{}{}
	return true
}

func (s *Walk) leave(key walkVisit) {
	if s.ptrLevel > startDetectingCyclesAfter {
		delete(s.seen, key)
	}
	s.ptrLevel--
}

// EnterPointer records the non-nil pointer p as an ancestor of the
// value being encoded. Past the pointer-depth threshold a repeat visit
// reports the cycle error of encoding/json.
func EnterPointer[P ~*T, T any](s *Walk, p P) error {
	if s.ptrLevel++; s.ptrLevel <= startDetectingCyclesAfter || s.visit(pointerVisit(p)) {
		return nil
	}
	return cycleError(reflect.ValueOf(p))
}

// LeavePointer closes an EnterPointer edge.
func LeavePointer[P ~*T, T any](s *Walk, p P) {
	s.leave(pointerVisit(p))
}

// EnterSlice records the non-nil slice v as an ancestor, keyed by its
// backing array and length, like EnterPointer.
func EnterSlice[S ~[]E, E any](s *Walk, v S) error {
	if s.ptrLevel++; s.ptrLevel <= startDetectingCyclesAfter || s.visit(sliceVisit(v)) {
		return nil
	}
	return cycleError(reflect.ValueOf(v))
}

// LeaveSlice closes an EnterSlice edge.
func LeaveSlice[S ~[]E, E any](s *Walk, v S) {
	s.leave(sliceVisit(v))
}

// EnterMap records the non-nil map m as an ancestor, like EnterPointer.
func EnterMap[M ~map[K]V, K comparable, V any](s *Walk, m M) error {
	if s.ptrLevel++; s.ptrLevel <= startDetectingCyclesAfter || s.visit(mapVisit(m)) {
		return nil
	}
	return cycleError(reflect.ValueOf(m))
}

// LeaveMap closes an EnterMap edge.
func LeaveMap[M ~map[K]V, K comparable, V any](s *Walk, m M) {
	s.leave(mapVisit(m))
}

func (s *Walk) enterValue(v reflect.Value) error {
	if s.ptrLevel++; s.ptrLevel <= startDetectingCyclesAfter || s.visit(valueVisit(v)) {
		return nil
	}
	return cycleError(v)
}

func (s *Walk) leaveValue(v reflect.Value) {
	s.leave(valueVisit(v))
}

func sliceData[S ~[]E, E any](v S) unsafe.Pointer {
	return unsafe.Pointer(unsafe.SliceData([]E(v)))
}

func mapData[M ~map[K]V, K comparable, V any](m M) unsafe.Pointer {
	return *(*unsafe.Pointer)(unsafe.Pointer(&m))
}

// AppendAny appends the encoding of v with this walk's state. Generated
// code calls it for a value whose dynamic type is known only at run time
// and for a value of a type it cannot spell. A value whose type Gen knows
// is encoded by generated code; any other value is walked the way
// encoding/json walks it, so cycles and the nesting depth stay counted
// across generated and dynamic values alike.
func (s *Walk) AppendAny(buf []byte, v any) ([]byte, error) {
	return s.appendValue(buf, reflect.ValueOf(v))
}

// AppendAddress appends the encoding of the value at the non-nil pointer
// p. The caller owns the pointer's visit edge; this entry does not add one.
func (s *Walk) AppendAddress(buf []byte, p any) ([]byte, error) {
	return s.appendValue(buf, reflect.ValueOf(p).Elem())
}

func (s *Walk) appendGen(buf []byte, rv reflect.Value) ([]byte, bool, error) {
	if s.Gen == nil || !rv.CanInterface() {
		return buf, false, nil
	}
	if k := rv.Kind(); k != reflect.Struct && k != reflect.Pointer {
		return buf, false, nil
	}
	return s.Gen(s, buf, rv.Interface())
}

func (s *Walk) appendValue(buf []byte, rv reflect.Value) ([]byte, error) {
	if !rv.IsValid() {
		return append(buf, "null"...), nil
	}
	if rv.Type() == numberType {
		return appendNumber(buf, rv)
	}
	if out, ok, err := s.appendGen(buf, rv); ok {
		return out, err
	}
	if hasJSONMethod(rv) {
		return appendHonored(buf, rv)
	}
	return s.appendKind(buf, rv)
}

func (s *Walk) appendKind(buf []byte, rv reflect.Value) ([]byte, error) {
	switch rv.Kind() {
	case reflect.Interface, reflect.Pointer:
		return s.appendIndirect(buf, rv)
	case reflect.Slice:
		return s.appendSlice(buf, rv)
	case reflect.Array:
		return s.appendArray(buf, rv)
	case reflect.Map:
		return s.appendMapValue(buf, rv)
	case reflect.Struct:
		return s.appendStruct(buf, rv)
	}
	return appendScalar(buf, rv)
}

func appendScalar(buf []byte, rv reflect.Value) ([]byte, error) {
	switch rv.Kind() {
	case reflect.Bool:
		return strconv.AppendBool(buf, rv.Bool()), nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return AppendInt(buf, rv.Int()), nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return AppendUint(buf, rv.Uint()), nil
	case reflect.Float32, reflect.Float64:
		return appendFloatValue(buf, rv)
	case reflect.String:
		buf = append(buf, '"')
		buf = AppendEscaped(buf, rv.String())
		return append(buf, '"'), nil
	}
	return buf, &stdjson.UnsupportedTypeError{Type: rv.Type()}
}

func appendFloatValue(buf []byte, rv reflect.Value) ([]byte, error) {
	bits := int(rv.Type().Size()) * 8
	f := rv.Float()
	if f-f != 0 {
		return buf, FloatErr(rv.Interface(), f, bits)
	}
	return AppendFloat(buf, f, bits), nil
}

func (s *Walk) appendIndirect(buf []byte, rv reflect.Value) ([]byte, error) {
	if rv.IsNil() {
		return append(buf, "null"...), nil
	}
	if rv.Kind() == reflect.Interface {
		return s.appendValue(buf, rv.Elem())
	}
	if err := s.enterValue(rv); err != nil {
		return buf, err
	}
	buf, err := s.appendValue(buf, rv.Elem())
	s.leaveValue(rv)
	return buf, err
}

func (s *Walk) appendSlice(buf []byte, rv reflect.Value) ([]byte, error) {
	if rv.IsNil() {
		return append(buf, "null"...), nil
	}
	if rv.Type().Elem().Kind() == reflect.Uint8 && !sliceElemHasJSONMethod(rv.Type().Elem()) {
		return AppendBase64(buf, rv.Bytes()), nil
	}
	if err := s.EnterElems(int64(len(buf)), rv.Len()); err != nil {
		return buf, err
	}
	defer s.LeaveFrame()
	if err := s.enterValue(rv); err != nil {
		return buf, err
	}
	buf, err := s.appendElems(buf, rv)
	s.leaveValue(rv)
	return buf, err
}

func (s *Walk) appendArray(buf []byte, rv reflect.Value) ([]byte, error) {
	if err := s.EnterFrame(int64(len(buf))); err != nil {
		return buf, err
	}
	defer s.LeaveFrame()
	return s.appendElems(buf, rv)
}

func (s *Walk) appendElems(buf []byte, rv reflect.Value) ([]byte, error) {
	buf = append(buf, '[')
	for i := range rv.Len() {
		if i > 0 {
			buf = append(buf, ',')
		}
		var err error
		if buf, err = s.appendValue(buf, rv.Index(i)); err != nil {
			return buf, err
		}
	}
	return append(buf, ']'), nil
}

func (s *Walk) appendMapValue(buf []byte, rv reflect.Value) ([]byte, error) {
	if !walkMapKeySupported(rv.Type().Key()) {
		return appendMarshaled(buf, rv.Interface())
	}
	if rv.IsNil() {
		return append(buf, "null"...), nil
	}
	if err := s.EnterElems(int64(len(buf)), rv.Len()); err != nil {
		return buf, err
	}
	defer s.LeaveFrame()
	if err := s.enterValue(rv); err != nil {
		return buf, err
	}
	buf, err := s.appendMap(buf, rv)
	s.leaveValue(rv)
	return buf, err
}

func (s *Walk) appendMap(buf []byte, rv reflect.Value) ([]byte, error) {
	es := make([]MapEntry[reflect.Value], 0, rv.Len())
	mi := rv.MapRange()
	if n := rv.Len(); JSONV2 && n >= 2 {
		for i := 0; i < n && mi.Next(); i++ {
			value := mi.Value()
			key, err := resolveWalkKey(mi.Key())
			if err != nil {
				return buf, keyError(rv.Type().String(), err)
			}
			es = append(es, MapEntry[reflect.Value]{key, value})
		}
	} else {
		for mi.Next() {
			key, err := resolveWalkKey(mi.Key())
			if err != nil {
				return buf, keyError(rv.Type().String(), err)
			}
			es = append(es, MapEntry[reflect.Value]{key, mi.Value()})
		}
	}
	buf = append(buf, '{')
	for i, e := range sortEntries(es) {
		if i > 0 {
			buf = append(buf, ',')
		}
		buf = append(buf, '"')
		buf = AppendEscaped(buf, e.Key)
		buf = append(buf, '"', ':')
		var err error
		if buf, err = s.appendValue(buf, e.Value); err != nil {
			return buf, err
		}
	}
	return append(buf, '}'), nil
}

func resolveWalkKey(k reflect.Value) (string, error) {
	if k.Kind() == reflect.String {
		return k.String(), nil
	}
	if k.Kind() == reflect.Interface {
		if k.IsNil() {
			return "", nilMapKeyError(k)
		}
		if text, ok := rawMapKeyText(k); ok {
			return text, nil
		}
	}
	if tm, ok := reflect.TypeAssert[encoding.TextMarshaler](k); ok {
		if k.Kind() == reflect.Pointer && k.IsNil() {
			return "", nil
		}
		text, err := tm.MarshalText()
		return string(text), err
	}
	switch k.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return strconv.FormatInt(k.Int(), 10), nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return strconv.FormatUint(k.Uint(), 10), nil
	}
	panic("unexpected map key type")
}

var numberType = reflect.TypeFor[stdjson.Number]()

func numberBody(rv reflect.Value) ([]byte, error) {
	return MarshalField(stdjson.Number(rv.String()))
}

func appendNumber(buf []byte, rv reflect.Value) ([]byte, error) {
	return appendMarshaled(buf, stdjson.Number(rv.String()))
}

func appendMarshaled(buf []byte, v any) ([]byte, error) {
	b, err := MarshalField(v)
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
	if hasJSONMethod(rv) {
		return appendHonored(buf, rv)
	}
	if rv.Kind() == reflect.Pointer {
		if rv.IsNil() {
			return append(buf, "null"...), nil
		}
		return s.appendStringOptionValue(buf, rv.Elem())
	}
	return s.appendQuotedScalar(buf, rv)
}

func (s *Walk) appendQuotedScalar(buf []byte, rv reflect.Value) ([]byte, error) {
	if rv.Kind() == reflect.String {
		buf = append(buf, '"')
		buf = AppendQuoted(buf, rv.String())
		return append(buf, '"'), nil
	}
	if !isQuotedWalkKind(rv.Kind()) {
		return s.appendValue(buf, rv)
	}
	buf, err := appendScalar(append(buf, '"'), rv)
	if err != nil {
		return buf, err
	}
	return append(buf, '"'), nil
}

func appendHonored(buf []byte, rv reflect.Value) ([]byte, error) {
	return appendMarshaled(buf, addrWalkInterface(rv))
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
		if elem.Implements(iface) || reflect.PointerTo(elem).Implements(iface) {
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

// AppendBase64 appends b as a JSON string holding its standard base64
// encoding, the encoding/json form of a byte slice.
func AppendBase64[S ~[]E, E ~uint8](dst []byte, b S) []byte {
	dst = append(dst, '"')
	dst = base64.StdEncoding.AppendEncode(dst, unsafe.Slice((*byte)(sliceData(b)), len(b)))
	return append(dst, '"')
}

// MapEntry is one map entry with its key in JSON text form.
type MapEntry[V any] struct {
	Key   string
	Value V
}

func sortEntries[V any](es []MapEntry[V]) []MapEntry[V] {
	slices.SortFunc(es, func(a, b MapEntry[V]) int { return strings.Compare(a.Key, b.Key) })
	return es
}

// StringEntries returns the entries of a string-keyed map sorted by key.
// The error is always nil; every entry builder shares one result shape.
func StringEntries[M ~map[K]V, K ~string, V any](m M) ([]MapEntry[V], error) {
	es := make([]MapEntry[V], 0, len(m))
	for k, v := range m {
		es = append(es, MapEntry[V]{string(k), v})
	}
	return sortEntries(es), nil
}

// IntEntries returns the entries of a signed-integer-keyed map sorted by
// the decimal key text. The error is always nil.
func IntEntries[M ~map[K]V, K ~int | ~int8 | ~int16 | ~int32 | ~int64, V any](m M) ([]MapEntry[V], error) {
	es := make([]MapEntry[V], 0, len(m))
	for k, v := range m {
		es = append(es, MapEntry[V]{strconv.FormatInt(int64(k), 10), v})
	}
	return sortEntries(es), nil
}

// UintEntries returns the entries of an unsigned-integer-keyed map sorted
// by the decimal key text. The error is always nil.
func UintEntries[M ~map[K]V, K ~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr, V any](m M) ([]MapEntry[V], error) {
	es := make([]MapEntry[V], 0, len(m))
	for k, v := range m {
		es = append(es, MapEntry[V]{strconv.FormatUint(uint64(k), 10), v})
	}
	return sortEntries(es), nil
}

// TextEntries returns the entries of a map whose non-pointer key type
// implements encoding.TextMarshaler, sorted by the key text.
func TextEntries[M ~map[K]V, K interface {
	comparable
	encoding.TextMarshaler
}, V any](m M, typeName string) ([]MapEntry[V], error) {
	es := make([]MapEntry[V], 0, len(m))
	var key K
	var value V
	kp := reflect.ValueOf(&key).Elem()
	vp := reflect.ValueOf(&value).Elem()
	mi := reflect.ValueOf(m).MapRange()
	if n := len(m); JSONV2 && n >= 2 {
		for i := 0; i < n && mi.Next(); i++ {
			kp.SetIterKey(mi)
			vp.SetIterValue(mi)
			text, err := textKeyText(key, kp)
			if err != nil {
				return nil, keyError(typeNameOr[M](typeName), err)
			}
			es = append(es, MapEntry[V]{text, value})
		}
	} else {
		for mi.Next() {
			kp.SetIterKey(mi)
			text, err := textKeyText(key, kp)
			if err != nil {
				return nil, keyError(typeNameOr[M](typeName), err)
			}
			vp.SetIterValue(mi)
			es = append(es, MapEntry[V]{text, value})
		}
	}
	return sortEntries(es), nil
}

// PointerTextEntries is TextEntries for a pointer key type; a nil key
// has the empty text.
func PointerTextEntries[M ~map[K]V, K interface {
	~*E
	encoding.TextMarshaler
}, E any, V any](m M, typeName string) ([]MapEntry[V], error) {
	es := make([]MapEntry[V], 0, len(m))
	var key K
	var value V
	kp := reflect.ValueOf(&key).Elem()
	vp := reflect.ValueOf(&value).Elem()
	mi := reflect.ValueOf(m).MapRange()
	if n := len(m); JSONV2 && n >= 2 {
		for i := 0; i < n && mi.Next(); i++ {
			kp.SetIterKey(mi)
			vp.SetIterValue(mi)
			text, err := pointerKeyText(key)
			if err != nil {
				return nil, keyError(typeNameOr[M](typeName), err)
			}
			es = append(es, MapEntry[V]{text, value})
		}
	} else {
		for mi.Next() {
			kp.SetIterKey(mi)
			text, err := pointerKeyText(key)
			if err != nil {
				return nil, keyError(typeNameOr[M](typeName), err)
			}
			vp.SetIterValue(mi)
			es = append(es, MapEntry[V]{text, value})
		}
	}
	return sortEntries(es), nil
}

func pointerKeyText[K interface {
	~*E
	encoding.TextMarshaler
}, E any](k K) (string, error) {
	if k == nil {
		return "", nil
	}
	text, err := k.MarshalText()
	return string(text), err
}

func textKeyText[K encoding.TextMarshaler](key K, kv reflect.Value) (string, error) {
	if any(key) == nil {
		return "", nilMapKeyError(kv)
	}
	if text, ok := rawMapKeyText(kv); ok {
		return text, nil
	}
	b, err := key.MarshalText()
	return string(b), err
}

func typeNameOr[M any](name string) string {
	if name != "" {
		return name
	}
	return reflect.TypeFor[M]().String()
}

func nilMapKeyError(k reflect.Value) error {
	return &stdjson.UnsupportedValueError{Value: k, Str: "nil map key"}
}

func rawMapKeyText(k reflect.Value) (string, bool) {
	if !JSONV2 {
		return "", false
	}
	if k.Kind() == reflect.Interface {
		k = k.Elem()
	}
	if k.Kind() != reflect.String {
		return "", false
	}
	return k.String(), true
}

func keyError(typeName string, err error) error {
	return fmt.Errorf("json: encoding error for type %q: %q", typeName, err.Error())
}
