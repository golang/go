// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package utf8_test

import (
	"rand"
	"testing"
	. "utf8"
)

func TestScanForwards(t *testing.T) {
	for _, s := range testStrings {
		runes := []int(s)
		str := NewString(s)
		if str.RuneCount() != len(runes) {
			t.Error("%s: expected %d runes; got %d", s, len(runes), str.RuneCount())
			break
		}
		for i, expect := range runes {
			got := str.At(i)
			if got != expect {
				t.Errorf("%s[%d]: expected %c (U+%04x); got %c (U+%04x)", s, i, expect, expect, got, got)
			}
		}
	}
}

func TestScanBackwards(t *testing.T) {
	for _, s := range testStrings {
		runes := []int(s)
		str := NewString(s)
		if str.RuneCount() != len(runes) {
			t.Error("%s: expected %d runes; got %d", s, len(runes), str.RuneCount())
			break
		}
		for i := len(runes) - 1; i >= 0; i-- {
			expect := runes[i]
			got := str.At(i)
			if got != expect {
				t.Errorf("%s[%d]: expected %c (U+%04x); got %c (U+%04x)", s, i, expect, expect, got, got)
			}
		}
	}
}

const randCount = 100000

func TestRandomAccess(t *testing.T) {
	for _, s := range testStrings {
		if len(s) == 0 {
			continue
		}
		runes := []int(s)
		str := NewString(s)
		if str.RuneCount() != len(runes) {
			t.Error("%s: expected %d runes; got %d", s, len(runes), str.RuneCount())
			break
		}
		for j := 0; j < randCount; j++ {
			i := rand.Intn(len(runes))
			expect := runes[i]
			got := str.At(i)
			if got != expect {
				t.Errorf("%s[%d]: expected %c (U+%04x); got %c (U+%04x)", s, i, expect, expect, got, got)
			}
		}
	}
}
