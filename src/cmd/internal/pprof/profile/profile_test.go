// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package profile

import (
	"bytes"
	"testing"
)

func TestEmptyProfile(t *testing.T) {
	var buf bytes.Buffer
	p, err := Parse(&buf)
	if err != nil {
		t.Error("Want no error, got", err)
	}
	if p == nil {
		t.Fatal("Want a valid profile, got <nil>")
	}
	if !p.Empty() {
		t.Errorf("Profile should be empty, got %#v", p)
	}
}
