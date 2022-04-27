// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build boringcrypto

package fipsonly

import (
	"crypto/internal/boring/fipstls"
	"testing"
)

func Test(t *testing.T) {
	if !fipstls.Required() {
		t.Fatal("fipstls.Required() = false, must be true")
	}
}
