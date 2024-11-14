// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package base

import (
	"bytes"
	"internal/bisect"
	"strings"
	"testing"
)

func TestHashDebugGossahashY(t *testing.T) {
	hd := NewHashDebug("GOSSAHASH", "y", new(bytes.Buffer))
	if hd == nil {
		t.Errorf("NewHashDebug should not return nil for GOSSASHASH=y")
	}
	if !hd.MatchPkgFunc("anything", "anyfunc", nil) {
		t.Errorf("NewHashDebug should return yes for everything for GOSSASHASH=y")
	}
}

func TestHashDebugGossahashN(t *testing.T) {
	hd := NewHashDebug("GOSSAHASH", "n", new(bytes.Buffer))
	if hd == nil {
		t.Errorf("NewHashDebug should not return nil for GOSSASHASH=n")
	}
	if hd.MatchPkgFunc("anything", "anyfunc", nil) {
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
	h0 := bisect.Hash("bar", "0")
	h1 := bisect.Hash("bar", "1")
	t.Logf(`These values are used in other tests: Hash("bar", "0")=%#64b, Hash("bar", "1")=%#64b`, h0, h1)
	if h0 == h1 {
		t.Errorf("Hashes 0x%x and 0x%x should differ", h0, h1)
	}
}

func TestHashMatch(t *testing.T) {
	b := new(bytes.Buffer)
	hd := NewHashDebug("GOSSAHASH", "v1110", b)
	check := hd.MatchPkgFunc("bar", "0", func { "note" })
	msg := b.String()
	t.Logf("message was '%s'", msg)
	if !check {
		t.Errorf("GOSSAHASH=1110 should have matched for 'bar', '0'")
	}
	wantPrefix(t, msg, "bar.0: note [bisect-match ")
	wantContains(t, msg, "\nGOSSAHASH triggered bar.0: note ")
}

func TestYMatch(t *testing.T) {
	b := new(bytes.Buffer)
	hd := NewHashDebug("GOSSAHASH", "vy", b)
	check := hd.MatchPkgFunc("bar", "0", nil)
	msg := b.String()
	t.Logf("message was '%s'", msg)
	if !check {
		t.Errorf("GOSSAHASH=y should have matched for 'bar', '0'")
	}
	wantPrefix(t, msg, "bar.0 [bisect-match ")
	wantContains(t, msg, "\nGOSSAHASH triggered bar.0 010100100011100101011110")
}

func TestNMatch(t *testing.T) {
	b := new(bytes.Buffer)
	hd := NewHashDebug("GOSSAHASH", "vn", b)
	check := hd.MatchPkgFunc("bar", "0", nil)
	msg := b.String()
	t.Logf("message was '%s'", msg)
	if check {
		t.Errorf("GOSSAHASH=n should NOT have matched for 'bar', '0'")
	}
	wantPrefix(t, msg, "bar.0 [DISABLED] [bisect-match ")
	wantContains(t, msg, "\nGOSSAHASH triggered bar.0 [DISABLED] 010100100011100101011110")
}

func TestHashNoMatch(t *testing.T) {
	b := new(bytes.Buffer)
	hd := NewHashDebug("GOSSAHASH", "01110", b)
	check := hd.MatchPkgFunc("bar", "0", nil)
	msg := b.String()
	t.Logf("message was '%s'", msg)
	if check {
		t.Errorf("GOSSAHASH=001100 should NOT have matched for 'bar', '0'")
	}
	if msg != "" {
		t.Errorf("Message should have been empty, instead %s", msg)
	}

}

func TestHashSecondMatch(t *testing.T) {
	b := new(bytes.Buffer)
	hd := NewHashDebug("GOSSAHASH", "01110/11110", b)

	check := hd.MatchPkgFunc("bar", "0", nil)
	msg := b.String()
	t.Logf("message was '%s'", msg)
	if !check {
		t.Errorf("GOSSAHASH=001100, GOSSAHASH0=0011 should have matched for 'bar', '0'")
	}
	wantContains(t, msg, "\nGOSSAHASH0 triggered bar")
}

func wantPrefix(t *testing.T, got, want string) {
	t.Helper()
	if !strings.HasPrefix(got, want) {
		t.Errorf("want prefix %q, got:\n%s", want, got)
	}
}

func wantContains(t *testing.T, got, want string) {
	t.Helper()
	if !strings.Contains(got, want) {
		t.Errorf("want contains %q, got:\n%s", want, got)
	}
}
