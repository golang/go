// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package web2

import (
	"reflect"
	"testing"
)

var testNetrc = `
machine api.github.com
  login user
  password pwd

machine incomlete.host
  login justlogin
  
machine test.host
login user2
password pwd2
`

func TestReadNetrc(t *testing.T) {
	lines := parseNetrc(testNetrc)
	want := []netrcLine{
		{"api.github.com", "user", "pwd"},
		{"test.host", "user2", "pwd2"},
	}

	if !reflect.DeepEqual(lines, want) {
		t.Errorf("parseNetrc:\nhave %q\nwant %q", lines, want)
	}
}
