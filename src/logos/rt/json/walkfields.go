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

type walkField struct {
	name      string
	tag       bool
	index     []int
	typ       reflect.Type
	omitEmpty bool
	omitZero  bool
	quoted    bool
	nameHTML  string
	isZero    func(reflect.Value) bool
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
	if s.fields == nil {
		s.fields = make(map[reflect.Type][]walkField)
	}
	if f, ok := s.fields[t]; ok {
		return f
	}
	f := walkFields(t)
	s.fields[t] = f
	return f
}

func (sc *walkScan) scanType(f walkType) {
	st := f.typ
	for i := range st.NumField() {
		sf := st.Field(i)
		ft := walkFieldType(sf)
		if ft == nil {
			continue
		}
		sc.considerField(f, sf, i, *ft)
	}
}

func walkFieldType(sf reflect.StructField) *reflect.Type {
	if sf.Anonymous {
		ft := sf.Type
		if ft.Name() == "" && ft.Kind() == reflect.Pointer {
			ft = ft.Elem()
		}
		if sf.PkgPath != "" && ft.Kind() != reflect.Struct {
			return nil
		}
	} else if sf.PkgPath != "" {
		return nil
	}
	if sf.Tag.Get("json") == "-" {
		return nil
	}
	ft := sf.Type
	if ft.Name() == "" && ft.Kind() == reflect.Pointer {
		ft = ft.Elem()
	}
	return &ft
}

func (sc *walkScan) considerField(f walkType, sf reflect.StructField, i int, ft reflect.Type) {
	name, opts := parseWalkTag(sf.Tag.Get("json"))
	if !isValidWalkTag(name) {
		name = ""
	}
	index := make([]int, len(f.index)+1)
	copy(index, f.index)
	index[len(f.index)] = i

	if name == "" && sf.Anonymous && ft.Kind() == reflect.Struct {
		sc.nextC[ft]++
		if sc.nextC[ft] == 1 {
			sc.next = append(sc.next, walkType{typ: ft, index: index})
		}
		return
	}
	tagged := name != ""
	if name == "" {
		name = sf.Name
	}
	field := walkField{
		name:      name,
		tag:       tagged,
		index:     index,
		typ:       ft,
		omitEmpty: walkTagOptions(opts).Contains("omitempty"),
		omitZero:  walkTagOptions(opts).Contains("omitzero"),
		quoted:    isQuotedWalkKind(ft.Kind()) && walkTagOptions(opts).Contains("string"),
	}
	buf := AppendEscaped(nil, field.name)
	field.nameHTML = `"` + string(buf) + `":`
	field.isZero = walkIsZero(sf.Type)
	sc.fields = append(sc.fields, field)
	if sc.count[f.typ] > 1 {
		sc.fields = append(sc.fields, sc.fields[len(sc.fields)-1])
	}
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

func walkIsZero(ft reflect.Type) func(reflect.Value) bool {
	if !fieldNeedsIsZero(ft) {
		return nil
	}
	switch {
	case ft.Kind() == reflect.Interface && ft.Implements(isZeroerType):
		return func(v reflect.Value) bool {
			return v.IsNil() ||
				(v.Elem().Kind() == reflect.Pointer && v.Elem().IsNil()) ||
				v.Interface().(isZeroer).IsZero()
		}
	case ft.Kind() == reflect.Pointer && ft.Implements(isZeroerType):
		return func(v reflect.Value) bool {
			return v.IsNil() || v.Interface().(isZeroer).IsZero()
		}
	case ft.Implements(isZeroerType):
		return func(v reflect.Value) bool {
			return v.Interface().(isZeroer).IsZero()
		}
	default:
		return func(v reflect.Value) bool {
			if !v.CanAddr() {
				v2 := reflect.New(v.Type()).Elem()
				v2.Set(v)
				v = v2
			}
			return v.Addr().Interface().(isZeroer).IsZero()
		}
	}
}

func fieldNeedsIsZero(ft reflect.Type) bool {
	return ft.Implements(isZeroerType) || reflect.PointerTo(ft).Implements(isZeroerType)
}

func walkFields(t reflect.Type) []walkField {
	return finishWalkFields(allWalkFields(t))
}

func allWalkFields(t reflect.Type) []walkField {
	next := []walkType{{typ: t}}
	var current []walkType
	var count, nextCount map[reflect.Type]int
	visited := map[reflect.Type]bool{}
	sc := &walkScan{}
	var done []walkField
	for len(next) > 0 {
		current, next = next, current[:0]
		count, nextCount = nextCount, map[reflect.Type]int{}
		sc.fields = nil
		sc.count = count
		sc.nextC = nextCount
		sc.next = next
		for _, f := range current {
			if visited[f.typ] {
				continue
			}
			visited[f.typ] = true
			sc.scanType(f)
		}
		next = sc.next
		nextCount = sc.nextC
		done = append(done, sc.fields...)
	}
	return done
}

func finishWalkFields(fields []walkField) []walkField {
	slices.SortFunc(fields, func(a, b walkField) int {
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
	})

	out := fields[:0]
	for i := 0; i < len(fields); {
		advance := 1
		for i+advance < len(fields) && fields[i+advance].name == fields[i].name {
			advance++
		}
		if advance == 1 {
			out = append(out, fields[i])
		} else if dominant, ok := dominantWalkField(fields[i : i+advance]); ok {
			out = append(out, dominant)
		}
		i += advance
	}

	fields = out
	slices.SortFunc(fields, func(a, b walkField) int {
		return slices.Compare(a.index, b.index)
	})
	return fields
}

func dominantWalkField(fields []walkField) (walkField, bool) {
	if len(fields) > 1 && len(fields[0].index) == len(fields[1].index) && fields[0].tag == fields[1].tag {
		return walkField{}, false
	}
	return fields[0], true
}

func parseWalkTag(tag string) (string, string) {
	name, opt, _ := strings.Cut(tag, ",")
	return name, opt
}

type walkTagOptions string

func (o walkTagOptions) Contains(optionName string) bool {
	for s := string(o); s != ""; {
		var name string
		name, s, _ = strings.Cut(s, ",")
		if name == optionName {
			return true
		}
	}
	return false
}

func isValidWalkTag(s string) bool {
	if s == "" {
		return false
	}
	for _, c := range s {
		switch {
		case strings.ContainsRune("!#$%&()*+-./:;<=>?@[]^_{|}~ ", c):
		case !unicode.IsLetter(c) && !unicode.IsDigit(c):
			return false
		}
	}
	return true
}

func (s *Walk) appendStruct(buf []byte, rv reflect.Value) ([]byte, error) {
	if err := s.EnterFrame(int64(len(buf))); err != nil {
		return buf, err
	}
	defer s.LeaveFrame()
	fields := s.walkFieldsOf(rv.Type())
	next := byte('{')
	for i := range fields {
		fv, ok := fieldByWalkIndex(rv, fields[i].index)
		if !ok {
			continue
		}
		f := &fields[i]
		if (f.omitEmpty && isEmptyWalkValue(fv)) ||
			(f.omitZero && (f.isZero == nil && fv.IsZero() || (f.isZero != nil && f.isZero(fv)))) {
			continue
		}
		buf = append(buf, next)
		next = ','
		buf = append(buf, f.nameHTML...)
		var err error
		if f.quoted {
			isNum := fv.Type() == numberType ||
				(fv.Kind() == reflect.Pointer && !fv.IsNil() && fv.Type().Elem() == numberType)
			if !isNum && s.honorsJSONMethod(fv) {
				buf, err = appendHonored(buf, fv)
			} else {
				buf, err = s.appendStringOptionValue(buf, fv)
			}
		} else {
			buf, err = s.appendValue(buf, fv)
		}
		if err != nil {
			return buf, err
		}
	}
	if next == '{' {
		return append(buf, "{}"...), nil
	}
	return append(buf, '}'), nil
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
