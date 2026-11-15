// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

import (
	"errors"
	"logos/arena"
	"reflect"
	"strconv"
	"strings"
	"testing"
)

func TestRtAppendMatchesStrconv(t *testing.T) {
	if got := string(AppendInt(nil, -12345, 10)); got != strconv.FormatInt(-12345, 10) {
		t.Fatalf("AppendInt = %q", got)
	}
	if got := string(AppendUint(nil, 12345, 10)); got != strconv.FormatUint(12345, 10) {
		t.Fatalf("AppendUint = %q", got)
	}
	if got := string(AppendFloat(nil, 1.5, 'g', -1, 64)); got != strconv.FormatFloat(1.5, 'g', -1, 64) {
		t.Fatalf("AppendFloat = %q", got)
	}
	if got := string(AppendBool(nil, true)); got != "true" {
		t.Fatalf("AppendBool true = %q", got)
	}
	if got := string(AppendBool(nil, false)); got != "false" {
		t.Fatalf("AppendBool false = %q", got)
	}
}

func TestRtHashDeterministic(t *testing.T) {
	if HashUint64(42) != HashUint64(42) {
		t.Fatal("HashUint64 not deterministic")
	}
	if HashBool(true) == HashBool(false) {
		t.Fatal("HashBool true equals false")
	}
	if HashFloat64(0.0) != HashFloat64(-0.0) {
		t.Fatal("HashFloat64 -0.0 differs from 0.0")
	}
	if HashCombine(1, 2) != HashCombine(1, 2) {
		t.Fatal("HashCombine not deterministic")
	}
	if ConstMix(7, 9) != ConstMix(7, 9) {
		t.Fatal("ConstMix not deterministic")
	}
	if HashUint64(1) == HashUint64(2) {
		t.Fatal("HashUint64 collision on 1 vs 2")
	}
}

func TestRtDeferTryJoins(t *testing.T) {
	var errp error
	DeferTry(&errp, nil)
	if errp != nil {
		t.Fatalf("DeferTry nil = %v", errp)
	}
	first := errors.New("first")
	DeferTry(&errp, first)
	if !errors.Is(errp, first) {
		t.Fatalf("DeferTry single = %v", errp)
	}
	second := errors.New("second")
	DeferTry(&errp, second)
	if !errors.Is(errp, first) || !errors.Is(errp, second) {
		t.Fatalf("DeferTry join = %v", errp)
	}
}

func TestRtOptionLookup(t *testing.T) {
	m := map[string][]byte{"a": nil}
	got := Lookup(m, "a")
	if !got.Present {
		t.Fatal("Lookup present nil payload not present")
	}
	if got.Value != nil {
		t.Fatalf("Lookup value = %v", got.Value)
	}
	miss := Lookup(m, "missing")
	if miss.Present {
		t.Fatal("Lookup missing present")
	}
	opt := Opt(7, true)
	if !opt.Present || opt.Value != 7 {
		t.Fatalf("Opt = %+v", opt)
	}
	if got := Bare(7, false); got != 0 {
		t.Fatalf("Bare absent = %d", got)
	}
}

func TestRtConstEntriesIterate(t *testing.T) {
	entries := []ConstEntry[string, int]{{Key: "b", Val: 2}, {Key: "a", Val: 1}}
	var keys []string
	for k, v := range ConstAll(entries) {
		keys = append(keys, k)
		if entries[0].Key == k && v != entries[0].Val {
			t.Fatalf("ConstAll value = %d", v)
		}
	}
	if !reflect.DeepEqual(keys, []string{"b", "a"}) {
		t.Fatalf("ConstAll order = %v", keys)
	}
}

func TestRtMissingFieldNamesType(t *testing.T) {
	err := MissingFieldError{Type: "Monster", Field: "HP"}
	msg := err.Error()
	if !strings.Contains(msg, "Monster") || !strings.Contains(msg, "HP") {
		t.Fatalf("MissingFieldError = %q", msg)
	}
}

func TestRtUnionTrialNamesOutcome(t *testing.T) {
	err := UnionTrialError{Union: "U", Members: []string{"A", "B"}, Matched: 2, Total: 2}
	msg := err.Error()
	if !strings.Contains(msg, "U") || !strings.Contains(msg, "A") || !strings.Contains(msg, "B") || !strings.Contains(msg, "2") {
		t.Fatalf("UnionTrialError ambiguous = %q", msg)
	}
	solo := UnionTrialError{Union: "U", Members: []string{"A"}, Matched: 0, Total: 1}
	if msg := solo.Error(); !strings.Contains(msg, "U") || !strings.Contains(msg, "0") || !strings.Contains(msg, "A") {
		t.Fatalf("UnionTrialError empty = %q", msg)
	}
}

func TestRtUnionAssertionNamesSides(t *testing.T) {
	err := &UnionAssertionError{Value: 7, Union: "pkg.U"}
	msg := err.Error()
	if !strings.Contains(msg, "int") || !strings.Contains(msg, "pkg.U") {
		t.Fatalf("UnionAssertionError = %q", msg)
	}
	nilErr := &UnionAssertionError{Value: nil, Union: "pkg.U"}
	if msg := nilErr.Error(); !strings.Contains(msg, "nil") || !strings.Contains(msg, "pkg.U") {
		t.Fatalf("UnionAssertionError nil = %q", msg)
	}
}

func TestRtRegionHeapEquivalent(t *testing.T) {
	var r Region
	p := arena.New[int](r)
	if p == nil {
		t.Fatal("New returned nil")
	}
	s := arena.Make[int](r, 3)
	if len(s) != 3 {
		t.Fatalf("Make len = %d", len(s))
	}
	s = arena.Append[int](r, s, 4, 5)
	if len(s) != 5 || s[3] != 4 || s[4] != 5 {
		t.Fatalf("Append = %v", s)
	}
	if got := RegionBytes(); got != 0 {
		t.Fatalf("RegionBytes = %d", got)
	}
}
