// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slog

import (
	"bytes"
	"flag"
	"strings"
	"testing"
)

func TestLevelString(t *testing.T) {
	for _, test := range []struct {
		in   Level
		want string
	}{
		{0, "INFO"},
		{LevelError, "ERROR"},
		{LevelError + 2, "ERROR+2"},
		{LevelError - 2, "WARN+2"},
		{LevelWarn, "WARN"},
		{LevelWarn - 1, "INFO+3"},
		{LevelInfo, "INFO"},
		{LevelInfo + 1, "INFO+1"},
		{LevelInfo - 3, "DEBUG+1"},
		{LevelDebug, "DEBUG"},
		{LevelDebug - 2, "DEBUG-2"},
	} {
		got := test.in.String()
		if got != test.want {
			t.Errorf("%d: got %s, want %s", test.in, got, test.want)
		}
	}
}

func TestLevelVar(t *testing.T) {
	var al LevelVar
	if got, want := al.Level(), LevelInfo; got != want {
		t.Errorf("got %v, want %v", got, want)
	}
	al.Set(LevelWarn)
	if got, want := al.Level(), LevelWarn; got != want {
		t.Errorf("got %v, want %v", got, want)
	}
	al.Set(LevelInfo)
	if got, want := al.Level(), LevelInfo; got != want {
		t.Errorf("got %v, want %v", got, want)
	}

}

func TestLevelMarshalJSON(t *testing.T) {
	want := LevelWarn - 3
	wantData := []byte(`"INFO+1"`)
	data, err := want.MarshalJSON()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(data, wantData) {
		t.Errorf("got %s, want %s", string(data), string(wantData))
	}
	var got Level
	if err := got.UnmarshalJSON(data); err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Errorf("got %s, want %s", got, want)
	}
}

func TestLevelMarshalText(t *testing.T) {
	want := LevelWarn - 3
	wantData := []byte("INFO+1")
	data, err := want.MarshalText()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(data, wantData) {
		t.Errorf("got %s, want %s", string(data), string(wantData))
	}
	var got Level
	if err := got.UnmarshalText(data); err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Errorf("got %s, want %s", got, want)
	}
}

func TestLevelAppendText(t *testing.T) {
	buf := make([]byte, 4, 16)
	want := LevelWarn - 3
	wantData := []byte("\x00\x00\x00\x00INFO+1")
	data, err := want.AppendText(buf)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(data, wantData) {
		t.Errorf("got %s, want %s", string(data), string(wantData))
	}
}

func TestLevelParse(t *testing.T) {
	for _, test := range []struct {
		in   string
		want Level
	}{
		{"DEBUG", LevelDebug},
		{"INFO", LevelInfo},
		{"WARN", LevelWarn},
		{"ERROR", LevelError},
		{"debug", LevelDebug},
		{"iNfo", LevelInfo},
		{"INFO+87", LevelInfo + 87},
		{"Error-18", LevelError - 18},
		{"Error-8", LevelInfo},
	} {
		var got Level
		if err := got.parse(test.in); err != nil {
			t.Fatalf("%q: %v", test.in, err)
		}
		if got != test.want {
			t.Errorf("%q: got %s, want %s", test.in, got, test.want)
		}
	}
}

func TestLevelParseError(t *testing.T) {
	for _, test := range []struct {
		in   string
		want string // error string should contain this
	}{
		{"", "unknown name"},
		{"dbg", "unknown name"},
		{"INFO+", "invalid syntax"},
		{"INFO-", "invalid syntax"},
		{"ERROR+23x", "invalid syntax"},
	} {
		var l Level
		err := l.parse(test.in)
		if err == nil || !strings.Contains(err.Error(), test.want) {
			t.Errorf("%q: got %v, want string containing %q", test.in, err, test.want)
		}
	}
}

func TestLevelFlag(t *testing.T) {
	fs := flag.NewFlagSet("test", flag.ContinueOnError)
	lf := LevelInfo
	fs.TextVar(&lf, "level", lf, "set level")
	err := fs.Parse([]string{"-level", "WARN+3"})
	if err != nil {
		t.Fatal(err)
	}
	if g, w := lf, LevelWarn+3; g != w {
		t.Errorf("got %v, want %v", g, w)
	}
}

func TestLevelVarMarshalText(t *testing.T) {
	var v LevelVar
	v.Set(LevelWarn)
	data, err := v.MarshalText()
	if err != nil {
		t.Fatal(err)
	}
	var v2 LevelVar
	if err := v2.UnmarshalText(data); err != nil {
		t.Fatal(err)
	}
	if g, w := v2.Level(), LevelWarn; g != w {
		t.Errorf("got %s, want %s", g, w)
	}
}

func TestLevelVarAppendText(t *testing.T) {
	var v LevelVar
	v.Set(LevelWarn)
	buf := make([]byte, 4, 16)
	data, err := v.AppendText(buf)
	if err != nil {
		t.Fatal(err)
	}
	var v2 LevelVar
	if err := v2.UnmarshalText(data[4:]); err != nil {
		t.Fatal(err)
	}
	if g, w := v2.Level(), LevelWarn; g != w {
		t.Errorf("got %s, want %s", g, w)
	}
}

func TestLevelVarFlag(t *testing.T) {
	fs := flag.NewFlagSet("test", flag.ContinueOnError)
	v := &LevelVar{}
	v.Set(LevelWarn + 3)
	fs.TextVar(v, "level", v, "set level")
	err := fs.Parse([]string{"-level", "WARN+3"})
	if err != nil {
		t.Fatal(err)
	}
	if g, w := v.Level(), LevelWarn+3; g != w {
		t.Errorf("got %v, want %v", g, w)
	}
}

func TestLevelVarString(t *testing.T) {
	var v LevelVar
	v.Set(LevelError)
	got := v.String()
	want := "LevelVar(ERROR)"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}
