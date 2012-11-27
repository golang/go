// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cookiejar

import (
	"testing"
)

var validStorageKeyTests = map[string]bool{
	"":            false,
	".":           false,
	"..":          false,
	"/":           false,
	"EXAMPLE.com": false,
	"\n":          false,
	"\r":          false,
	"\r\n":        false,
	"\x00":        false,
	"back\\slash": false,
	"co:lon":      false,
	"com,ma":      false,
	"semi;colon":  false,
	"sl/ash":      false,
	"sp ace":      false,
	"under_score": false,
	"π":           false,

	"-":                true,
	".dot":             true,
	".dot.":            true,
	".metadata":        true,
	".x..y..z...":      true,
	"dot.":             true,
	"example.com":      true,
	"foo":              true,
	"hy-phen":          true,
	"xn--bcher-kva.ch": true,
}

func TestValidStorageKey(t *testing.T) {
	for key, want := range validStorageKeyTests {
		if got := ValidStorageKey(key); got != want {
			t.Errorf("%q: got %v, want %v", key, got, want)
		}
	}
}
