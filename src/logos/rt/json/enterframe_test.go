// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	stdjson "encoding/json"
	"errors"
	"reflect"
	"sort"
	"strings"
	"testing"
)

func TestWalkEnterFrameBalances(t *testing.T) {
	var s Walk
	for range 100 {
		if err := s.EnterFrame(0); err != nil {
			t.Fatalf("EnterFrame = %v", err)
		}
	}
	for range 100 {
		s.LeaveFrame()
	}
	if err := s.EnterFrame(0); err != nil {
		t.Fatalf("EnterFrame after Leave = %v", err)
	}
	s.LeaveFrame()
}

func nestedArrays(depth int) nested {
	var v nested
	for range depth {
		v = nested{v}
	}
	return v
}

func TestWalkEnterFrameDepthBound(t *testing.T) {
	if !stdIsV2 {
		t.Skip("encoding/json v1 has no depth bound")
	}
	const searchLimit = 20000
	rejected := sort.Search(searchLimit, func(depth int) bool {
		_, err := stdjson.Marshal(nestedArrays(depth))
		return err != nil
	})
	if rejected == searchLimit {
		t.Fatalf("encoding/json accepted %d nested arrays", searchLimit)
	}
	accepted := rejected - 1
	var s Walk
	for pass := range 2 {
		for depth := range accepted {
			if err := s.EnterFrame(int64(depth)); err != nil {
				t.Fatalf("pass %d: EnterFrame failed at depth %d, encoding/json accepts %d: %v", pass, depth, accepted, err)
			}
		}
		if err := s.EnterFrame(int64(accepted)); err == nil {
			t.Fatalf("pass %d: EnterFrame accepted depth %d, encoding/json rejects it", pass, rejected)
		}
		for range accepted {
			s.LeaveFrame()
		}
	}
}

func TestRebaseLeavesUserSyntaxErrorUntouched(t *testing.T) {
	var v any
	err := stdjson.Unmarshal([]byte("{"), &v)
	user := err.(*stdjson.SyntaxError)
	before := *user
	var s Walk
	if got := s.RebaseDepthError(user, 71); got != user {
		t.Fatalf("user error identity changed: %p vs %p", got, user)
	}
	if user.Offset != before.Offset || user.Error() != before.Error() {
		t.Fatalf("user error was rebased: %#v vs %#v", user, &before)
	}
}

func TestWalkEnterVisitsAreBalanced(t *testing.T) {
	var s Walk
	p := &struct{ X int }{X: 1}
	m := map[string]int{"a": 1}
	sl := []int{1, 2}
	for range 2000 {
		if err := EnterPointer(&s, p); err != nil {
			t.Fatalf("EnterPointer = %v", err)
		}
		LeavePointer(&s, p)
		if err := EnterMap(&s, m); err != nil {
			t.Fatalf("EnterMap = %v", err)
		}
		LeaveMap(&s, m)
		if err := EnterSlice(&s, sl); err != nil {
			t.Fatalf("EnterSlice = %v", err)
		}
		LeaveSlice(&s, sl)
	}
	if s.ptrLevel != 0 || len(s.seen) != 0 {
		t.Fatalf("balanced visits left level %d and %d seen keys", s.ptrLevel, len(s.seen))
	}
}

func enterRepeated[V any](t *testing.T, label string, v V, enter func(*Walk, V) error, leave func(*Walk, V)) {
	t.Helper()
	var s Walk
	for range startDetectingCyclesAfter + 1 {
		if err := enter(&s, v); err != nil {
			t.Fatalf("%s: repeat setup = %v", label, err)
		}
	}
	err := enter(&s, v)
	unsupported, ok := errors.AsType[*stdjson.UnsupportedValueError](err)
	if !ok || !strings.HasPrefix(unsupported.Str, "encountered a cycle via ") {
		t.Fatalf("%s: repeat past threshold = %v, want the encoding/json cycle error", label, err)
	}
	if got := unsupported.Value.Type(); got != reflect.TypeOf(v) {
		t.Fatalf("%s: cycle error value type %v, want %T", label, got, v)
	}
	for range startDetectingCyclesAfter + 1 {
		leave(&s, v)
	}
	if err := enter(&s, v); err != nil {
		t.Fatalf("%s: enter after leaving every visit = %v", label, err)
	}
	leave(&s, v)
}

func TestWalkEnterCycleNeedsRepeat(t *testing.T) {
	enterRepeated(t, "pointer", &struct{ X int }{X: 2}, EnterPointer[*struct{ X int }], LeavePointer[*struct{ X int }])
	enterRepeated(t, "map", map[string]int{"a": 1}, EnterMap[map[string]int], LeaveMap[map[string]int])
	enterRepeated(t, "slice", []int{1, 2}, EnterSlice[[]int], LeaveSlice[[]int])
}

func TestWalkEnterSliceKeysOnLength(t *testing.T) {
	var s Walk
	backing := []int{1, 2, 3}
	for range startDetectingCyclesAfter {
		if err := EnterSlice(&s, []int{0}); err != nil {
			t.Fatalf("setup = %v", err)
		}
	}
	if err := EnterSlice(&s, backing); err != nil {
		t.Fatalf("first visit = %v", err)
	}
	if err := EnterSlice(&s, backing[:2]); err != nil {
		t.Fatalf("shorter view of the same backing array reported a cycle: %v", err)
	}
	if err := EnterSlice(&s, backing); err == nil {
		t.Fatal("same backing array and length did not report a cycle")
	}
}

type genT struct{ N int }

func TestWalkAppendAnyDispatch(t *testing.T) {
	var s Walk
	for _, v := range []any{nil, 1, "a<b>", []int{1}, map[string]any{"k": []any{"x&y"}}, genT{N: 1}} {
		got, err := s.AppendAny([]byte("["), v)
		want, werr := stdjson.Marshal(v)
		if err != nil || werr != nil || string(got) != "["+string(want) {
			t.Fatalf("AppendAny(%#v) = %q, %v; encoding/json %q, %v", v, got, err, want, werr)
		}
	}
	s.Gen = func(s *Walk, buf []byte, v any) ([]byte, bool, error) {
		if g, ok := v.(genT); ok {
			return AppendInt(buf, int64(g.N)), true, nil
		}
		return buf, false, nil
	}
	if got, err := s.AppendAny(nil, genT{N: 7}); err != nil || string(got) != "7" {
		t.Fatalf("AppendAny with Gen = %q, %v; want 7", got, err)
	}
	if got, err := s.AppendAny(nil, "x"); err != nil || string(got) != `"x"` {
		t.Fatalf("AppendAny outside Gen = %q, %v", got, err)
	}
}

type textKey int

func (k textKey) MarshalText() ([]byte, error) {
	if k < 0 {
		return nil, errors.New("negative key")
	}
	return []byte{'k', byte('a' + k)}, nil
}

type ptrKey struct{ n int }

func (k *ptrKey) MarshalText() ([]byte, error) {
	return []byte{'p', byte('a' + k.n)}, nil
}

func entryKeys[V any](es []MapEntry[V], err error) string {
	if err != nil {
		return "error: " + err.Error()
	}
	keys := make([]string, len(es))
	for i, e := range es {
		keys[i] = e.Key
	}
	return strings.Join(keys, ",")
}

func TestMapEntriesSortByKeyText(t *testing.T) {
	cases := []struct {
		got, want string
	}{
		{entryKeys(StringEntries(map[string]int{"b": 1, "a": 2, "": 3})), ",a,b"},
		{entryKeys(IntEntries(map[int8]int{10: 1, 2: 2, -1: 3})), "-1,10,2"},
		{entryKeys(UintEntries(map[uintptr]int{10: 1, 2: 2})), "10,2"},
		{entryKeys(TextEntries(map[textKey]int{2: 1, 0: 2})), "ka,kc"},
		{entryKeys(PointerTextEntries(map[*ptrKey]int{{n: 1}: 1, nil: 2})), ",pb"},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Errorf("entries = %q, want %q", c.got, c.want)
		}
	}
	m := map[textKey]int{-1: 1}
	_, err := TextEntries(m)
	_, werr := stdjson.Marshal(m)
	if err == nil || werr == nil {
		t.Fatalf("failing MarshalText key: entries error %v, encoding/json error %v", err, werr)
	}
	if !stdIsV2 && err.Error() != werr.Error() {
		t.Fatalf("failing MarshalText key: entries error %q, encoding/json error %q", err.Error(), werr.Error())
	}
}

func TestAppendBase64(t *testing.T) {
	type named uint8
	for _, b := range [][]byte{{}, []byte("bytes<&>"), {0, 1, 2, 255}} {
		want, err := stdjson.Marshal(b)
		if err != nil {
			t.Fatal(err)
		}
		if got := AppendBase64(nil, b); string(got) != string(want) {
			t.Errorf("AppendBase64(%v) = %s, want %s", b, got, want)
		}
	}
	if got := AppendBase64(nil, []named{1, 2}); string(got) != `"AQI="` {
		t.Errorf("AppendBase64 of a named byte slice = %s", got)
	}
}
