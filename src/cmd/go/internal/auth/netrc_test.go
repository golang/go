// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package auth

import (
	"reflect"
	"testing"
)

var testNetrc = `
machine incomplete
password none

machine api.github.com
  login user
  password pwd

machine incomlete.host
  login justlogin

machine test.host
login user2
password pwd2

machine oneline login user3 password pwd3

machine ignore.host macdef ignore
  login nobody
  password nothing

machine hasmacro.too macdef ignore-next-lines login user4 password pwd4
  login nobody
  password nothing

default
login anonymous
password gopher@golang.org

machine after.default
login oops
password too-late-in-file
`

func TestParseNetrc(t *testing.T) {
	lines := parseNetrc(testNetrc)
	want := []netrcLine{
		{"api.github.com", "user", "pwd"},
		{"test.host", "user2", "pwd2"},
		{"oneline", "user3", "pwd3"},
		{"hasmacro.too", "user4", "pwd4"},
	}

	if !reflect.DeepEqual(lines, want) {
		t.Errorf("parseNetrc:\nhave %q\nwant %q", lines, want)
	}
}
