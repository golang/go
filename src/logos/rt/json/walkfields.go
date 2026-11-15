// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"cmp"
	"reflect"
	"slices"
	"strings"
	"unicode"
)

type isZeroer interface {
	IsZero() bool
}

var isZeroerType = reflect.TypeFor[isZeroer]()

type walkField struct {
	name       string
	tag        bool
	index      []int
	omitEmpty  bool
	omitZero   bool
	addrIsZero bool
	quoted     bool
	nameHTML   string
	isZero     func(reflect.Value) bool
}

type walkType struct {
	typ   reflect.Type
	index []int
}

type walkScan struct {
	fields []walkField
	next   []walkType
	count  map[reflect.Type]int
	nextC  map[reflect.Type]int
}

func (s *Walk) walkFieldsOf(t reflect.Type) []walkField {
	if f, ok := s.fields[t]; ok {
		return f
	}
	if s.fields == nil {
		s.fields = make(map[reflect.Type][]walkField)
	}
	f := finishWalkFields(allWalkFields(t))
	s.fields[t] = f
	return f
}

func (sc *walkScan) scanType(f walkType) {
	for i := range f.typ.NumField() {
		sf := f.typ.Field(i)
		if ft, ok := walkFieldType(sf); ok {
			sc.considerField(f, sf, i, ft)
		}
	}
}

func derefUnnamed(t reflect.Type) reflect.Type {
	if t.Name() == "" && t.Kind() == reflect.Pointer {
		return t.Elem()
	}
	return t
}

func walkFieldType(sf reflect.StructField) (reflect.Type, bool) {
	ft := derefUnnamed(sf.Type)
	if !sf.IsExported() && (!sf.Anonymous || ft.Kind() != reflect.Struct) {
		return nil, false
	}
	return ft, sf.Tag.Get("json") != "-"
}

func (sc *walkScan) considerField(f walkType, sf reflect.StructField, i int, ft reflect.Type) {
	name, opts, _ := strings.Cut(sf.Tag.Get("json"), ",")
	if !isValidWalkTag(name) {
		name = ""
	}
	index := append(slices.Clone(f.index), i)
	if name == "" && sf.Anonymous && ft.Kind() == reflect.Struct {
		sc.nextC[ft]++
		if sc.nextC[ft] == 1 {
			sc.next = append(sc.next, walkType{typ: ft, index: index})
		}
		return
	}
	isZero, addrIsZero := walkIsZero(sf.Type)
	field := walkField{
		name:       cmp.Or(name, sf.Name),
		tag:        name != "",
		index:      index,
		omitEmpty:  walkTagHas(opts, "omitempty"),
		omitZero:   walkTagHas(opts, "omitzero"),
		addrIsZero: addrIsZero,
		quoted:     isQuotedWalkKind(ft.Kind()) && walkTagHas(opts, "string"),
		nameHTML:   `"` + string(AppendEscaped(nil, cmp.Or(name, sf.Name))) + `":`,
		isZero:     isZero,
	}
	sc.fields = append(sc.fields, field)
	if sc.count[f.typ] > 1 {
		sc.fields = append(sc.fields, field)
	}
}

func walkTagHas(opts, name string) bool {
	return slices.Contains(strings.Split(opts, ","), name)
}

func isQuotedWalkKind(k reflect.Kind) bool {
	switch k {
	case reflect.Bool,
		reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr,
		reflect.Float32, reflect.Float64,
		reflect.String:
		return true
	}
	return false
}

func walkIsZero(ft reflect.Type) (func(reflect.Value) bool, bool) {
	switch {
	case !ft.Implements(isZeroerType) && !reflect.PointerTo(ft).Implements(isZeroerType):
		return reflect.Value.IsZero, false
	case ft.Kind() == reflect.Interface && ft.Implements(isZeroerType):
		return func(v reflect.Value) bool {
			return v.IsNil() || (v.Elem().Kind() == reflect.Pointer && v.Elem().IsNil()) || v.Interface().(isZeroer).IsZero()
		}, false
	case ft.Kind() == reflect.Pointer && ft.Implements(isZeroerType):
		return func(v reflect.Value) bool {
			return v.IsNil() || v.Interface().(isZeroer).IsZero()
		}, false
	case ft.Implements(isZeroerType):
		return func(v reflect.Value) bool {
			return v.Interface().(isZeroer).IsZero()
		}, false
	}
	return func(v reflect.Value) bool {
		if !v.CanAddr() {
			v2 := reflect.New(v.Type()).Elem()
			v2.Set(v)
			v = v2
		}
		return v.Addr().Interface().(isZeroer).IsZero()
	}, true
}

func allWalkFields(t reflect.Type) []walkField {
	next := []walkType{{typ: t}}
	var count map[reflect.Type]int
	visited := map[reflect.Type]bool{}
	var done []walkField
	for len(next) > 0 {
		sc := &walkScan{count: count, nextC: map[reflect.Type]int{}}
		for _, f := range next {
			if !visited[f.typ] {
				visited[f.typ] = true
				sc.scanType(f)
			}
		}
		done = append(done, sc.fields...)
		next, count = sc.next, sc.nextC
	}
	return done
}

func compareWalkFields(a, b walkField) int {
	if c := strings.Compare(a.name, b.name); c != 0 {
		return c
	}
	if c := cmp.Compare(len(a.index), len(b.index)); c != 0 {
		return c
	}
	if a.tag != b.tag {
		if a.tag {
			return -1
		}
		return +1
	}
	return slices.Compare(a.index, b.index)
}

func finishWalkFields(fields []walkField) []walkField {
	slices.SortFunc(fields, compareWalkFields)
	var out []walkField
	for i := 0; i < len(fields); {
		n := 1
		for i+n < len(fields) && fields[i+n].name == fields[i].name {
			n++
		}
		if n == 1 || len(fields[i].index) != len(fields[i+1].index) || fields[i].tag != fields[i+1].tag {
			out = append(out, fields[i])
		}
		i += n
	}
	slices.SortFunc(out, func(a, b walkField) int { return slices.Compare(a.index, b.index) })
	return out
}

func isValidWalkTag(s string) bool {
	for _, c := range s {
		if !strings.ContainsRune("!#$%&()*+-./:;<=>?@[]^_{|}~ ", c) && !unicode.IsLetter(c) && !unicode.IsDigit(c) {
			return false
		}
	}
	return s != ""
}

func (s *Walk) appendStruct(buf []byte, rv reflect.Value) ([]byte, error) {
	if err := s.EnterFrame(int64(len(buf))); err != nil {
		return buf, err
	}
	defer s.LeaveFrame()
	next := byte('{')
	fields := s.walkFieldsOf(rv.Type())
	for i := range fields {
		f := &fields[i]
		fv, ok := fieldByWalkIndex(rv, f.index)
		if !ok {
			continue
		}
		copied := false
		if JSONV2 && f.omitZero && f.addrIsZero && !fv.CanAddr() {
			snapshot := reflect.New(fv.Type()).Elem()
			snapshot.Set(fv)
			fv = snapshot
			copied = true
		}
		if f.omitted(fv) {
			continue
		}
		if copied {
			fv = reflect.ValueOf(fv.Interface())
		}
		buf = append(append(buf, next), f.nameHTML...)
		next = ','
		var err error
		if buf, err = s.appendFieldValue(buf, f, fv); err != nil {
			return buf, err
		}
	}
	if next == '{' {
		buf = append(buf, '{')
	}
	return append(buf, '}'), nil
}

func (f *walkField) omitted(fv reflect.Value) bool {
	return (f.omitEmpty && isEmptyWalkValue(fv)) || (f.omitZero && f.isZero(fv))
}

func (s *Walk) appendFieldValue(buf []byte, f *walkField, fv reflect.Value) ([]byte, error) {
	if f.quoted {
		return s.appendStringOptionValue(buf, fv)
	}
	return s.appendValue(buf, fv)
}

func fieldByWalkIndex(rv reflect.Value, index []int) (reflect.Value, bool) {
	for _, i := range index {
		if rv.Kind() == reflect.Pointer {
			if rv.IsNil() {
				return rv, false
			}
			rv = rv.Elem()
		}
		rv = rv.Field(i)
	}
	return rv, true
}

func isEmptyWalkValue(v reflect.Value) bool {
	switch v.Kind() {
	case reflect.Array, reflect.Map, reflect.Slice, reflect.String:
		return v.Len() == 0
	case reflect.Bool,
		reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr,
		reflect.Float32, reflect.Float64,
		reflect.Interface, reflect.Pointer:
		return v.IsZero()
	}
	return false
}
