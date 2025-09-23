// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (cgo || darwin) && !osusergo && unix && !android

package user

import (
	"testing"
)

// Issue 22739
func TestNegativeUid(t *testing.T) {
	sp := structPasswdForNegativeTest()
	u := buildUser(&sp)
	if g, w := u.Uid, "4294967294"; g != w {
		t.Errorf("Uid = %q; want %q", g, w)
	}
	if g, w := u.Gid, "4294967293"; g != w {
		t.Errorf("Gid = %q; want %q", g, w)
	}
}
