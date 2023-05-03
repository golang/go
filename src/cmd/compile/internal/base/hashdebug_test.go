// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package base

import (
	"bytes"
	"strings"
	"testing"
)

func TestHashDebugGossahashY(t *testing.T) {
	hd := NewHashDebug("GOSSAHASH", "y", new(bufferWithSync))
	if hd == nil {
		t.Errorf("NewHashDebug should not return nil for GOSSASHASH=y")
	}
	if !hd.DebugHashMatch("anything") {
		t.Errorf("NewHashDebug should return yes for everything for GOSSASHASH=y")
	}
}

func TestHashDebugGossahashN(t *testing.T) {
	hd := NewHashDebug("GOSSAHASH", "n", new(bufferWithSync))
	if hd == nil {
		t.Errorf("NewHashDebug should not return nil for GOSSASHASH=n")
	}
	if hd.DebugHashMatch("anything") {
		t.Errorf("NewHashDebug should return no for everything for GOSSASHASH=n")
	}
}

func TestHashDebugGossahashEmpty(t *testing.T) {
	hd := NewHashDebug("GOSSAHASH", "", nil)
	if hd != nil {
		t.Errorf("NewHashDebug should return nil for GOSSASHASH=\"\"")
	}
}

func TestHashDebugMagic(t *testing.T) {
	hd := NewHashDebug("FOOXYZZY", "y", nil)
	hd0 := NewHashDebug("FOOXYZZY0", "n", nil)
	if hd == nil {
		t.Errorf("NewHashDebug should have succeeded for FOOXYZZY")
	}
	if hd0 == nil {
		t.Errorf("NewHashDebug should have succeeded for FOOXYZZY0")
	}
}

func TestHash(t *testing.T) {
	h0 := hashOf("bar", 0)
	h1 := hashOf("bar", 1)
	t.Logf(`These values are used in other tests: hashOf("bar,0)"=0x%x, hashOf("bar,1)"=0x%x`, h0, h1)
	if h0 == h1 {
		t.Errorf("Hashes 0x%x and 0x%x should differ", h0, h1)
	}
}

func TestHashMatch(t *testing.T) {
	ws := new(bufferWithSync)
	hd := NewHashDebug("GOSSAHASH", "0011", ws)
	check := hd.DebugHashMatch("bar")
	msg := ws.String()
	t.Logf("message was '%s'", msg)
	if !check {
		t.Errorf("GOSSAHASH=0011 should have matched for 'bar'")
	}
	wantPrefix(t, msg, "GOSSAHASH triggered bar ")
	wantContains(t, msg, "\nbar [bisect-match ")
}

func TestHashMatchParam(t *testing.T) {
	ws := new(bufferWithSync)
	hd := NewHashDebug("GOSSAHASH", "1010", ws)
	check := hd.DebugHashMatchParam("bar", 1)
	msg := ws.String()
	t.Logf("message was '%s'", msg)
	if !check {
		t.Errorf("GOSSAHASH=1010 should have matched for 'bar', 1")
	}
	wantPrefix(t, msg, "GOSSAHASH triggered bar:1 ")
	wantContains(t, msg, "\nbar:1 [bisect-match ")
}

func TestYMatch(t *testing.T) {
	ws := new(bufferWithSync)
	hd := NewHashDebug("GOSSAHASH", "y", ws)
	check := hd.DebugHashMatch("bar")
	msg := ws.String()
	t.Logf("message was '%s'", msg)
	if !check {
		t.Errorf("GOSSAHASH=y should have matched for 'bar'")
	}
	wantPrefix(t, msg, "GOSSAHASH triggered bar 110101000010000100000011")
	wantContains(t, msg, "\nbar [bisect-match ")
}

func TestNMatch(t *testing.T) {
	ws := new(bufferWithSync)
	hd := NewHashDebug("GOSSAHASH", "n", ws)
	check := hd.DebugHashMatch("bar")
	msg := ws.String()
	t.Logf("message was '%s'", msg)
	if check {
		t.Errorf("GOSSAHASH=n should NOT have matched for 'bar'")
	}
	wantPrefix(t, msg, "GOSSAHASH triggered bar 110101000010000100000011")
	wantContains(t, msg, "\nbar [bisect-match ")
}

func TestHashNoMatch(t *testing.T) {
	ws := new(bufferWithSync)
	hd := NewHashDebug("GOSSAHASH", "001100", ws)
	check := hd.DebugHashMatch("bar")
	msg := ws.String()
	t.Logf("message was '%s'", msg)
	if check {
		t.Errorf("GOSSAHASH=001100 should NOT have matched for 'bar'")
	}
	if msg != "" {
		t.Errorf("Message should have been empty, instead %s", msg)
	}

}

func TestHashSecondMatch(t *testing.T) {
	ws := new(bufferWithSync)
	hd := NewHashDebug("GOSSAHASH", "001100/0011", ws)

	check := hd.DebugHashMatch("bar")
	msg := ws.String()
	t.Logf("message was '%s'", msg)
	if !check {
		t.Errorf("GOSSAHASH=001100, GOSSAHASH0=0011 should have matched for 'bar'")
	}
	wantPrefix(t, msg, "GOSSAHASH0 triggered bar")
}

type bufferWithSync struct {
	b bytes.Buffer
}

func (ws *bufferWithSync) Sync() error {
	return nil
}

func (ws *bufferWithSync) Write(p []byte) (n int, err error) {
	return (&ws.b).Write(p)
}

func (ws *bufferWithSync) String() string {
	return strings.TrimSpace((&ws.b).String())
}

func wantPrefix(t *testing.T, got, want string) {
	if !strings.HasPrefix(got, want) {
		t.Errorf("want prefix %q, got:\n%s", want, got)
	}
}

func wantContains(t *testing.T, got, want string) {
	if !strings.Contains(got, want) {
		t.Errorf("want contains %q, got:\n%s", want, got)
	}
}
