// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"testing"
)

var strings = map[string]string{
	"\x20\x0d\x0a":  "\x20\x0d\x0a",
	"flibble":       "flibble",
	"new\x20line":   "new\x20line",
	"123456\x07789": "123456 789",
	"\t\t\x10\r\n":  "\t\t \r\n",
}

func TestSafeString(t *testing.T) {
	for s, expected := range strings {
		actual := safeString(s)
		if expected != actual {
			t.Errorf("expected: %v, actual: %v", []byte(expected), []byte(actual))
		}
	}
}
