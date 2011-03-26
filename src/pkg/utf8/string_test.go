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
			t.Errorf("%s: expected %d runes; got %d", s, len(runes), str.RuneCount())
			break
		}
		for i, expect := range runes {
			got := str.At(i)
			if got != expect {
				t.Errorf("%s[%d]: expected %c (%U); got %c (%U)", s, i, expect, expect, got, got)
			}
		}
	}
}

func TestScanBackwards(t *testing.T) {
	for _, s := range testStrings {
		runes := []int(s)
		str := NewString(s)
		if str.RuneCount() != len(runes) {
			t.Errorf("%s: expected %d runes; got %d", s, len(runes), str.RuneCount())
			break
		}
		for i := len(runes) - 1; i >= 0; i-- {
			expect := runes[i]
			got := str.At(i)
			if got != expect {
				t.Errorf("%s[%d]: expected %c (%U); got %c (%U)", s, i, expect, expect, got, got)
			}
		}
	}
}

func randCount() int {
	if testing.Short() {
		return 100
	}
	return 100000
}

func TestRandomAccess(t *testing.T) {
	for _, s := range testStrings {
		if len(s) == 0 {
			continue
		}
		runes := []int(s)
		str := NewString(s)
		if str.RuneCount() != len(runes) {
			t.Errorf("%s: expected %d runes; got %d", s, len(runes), str.RuneCount())
			break
		}
		for j := 0; j < randCount(); j++ {
			i := rand.Intn(len(runes))
			expect := runes[i]
			got := str.At(i)
			if got != expect {
				t.Errorf("%s[%d]: expected %c (%U); got %c (%U)", s, i, expect, expect, got, got)
			}
		}
	}
}

func TestRandomSliceAccess(t *testing.T) {
	for _, s := range testStrings {
		if len(s) == 0 || s[0] == '\x80' { // the bad-UTF-8 string fools this simple test
			continue
		}
		runes := []int(s)
		str := NewString(s)
		if str.RuneCount() != len(runes) {
			t.Errorf("%s: expected %d runes; got %d", s, len(runes), str.RuneCount())
			break
		}
		for k := 0; k < randCount(); k++ {
			i := rand.Intn(len(runes))
			j := rand.Intn(len(runes) + 1)
			if i > j { // include empty strings
				continue
			}
			expect := string(runes[i:j])
			got := str.Slice(i, j)
			if got != expect {
				t.Errorf("%s[%d:%d]: expected %q got %q", s, i, j, expect, got)
			}
		}
	}
}

func TestLimitSliceAccess(t *testing.T) {
	for _, s := range testStrings {
		str := NewString(s)
		if str.Slice(0, 0) != "" {
			t.Error("failure with empty slice at beginning")
		}
		nr := RuneCountInString(s)
		if str.Slice(nr, nr) != "" {
			t.Error("failure with empty slice at end")
		}
	}
}
