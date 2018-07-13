// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !go1.11

package packages_test

import (
	"testing"

	"golang.org/x/tools/go/packages"
)

func TestGoIsTooOld(t *testing.T) {
	_, err := packages.Metadata(nil, "errors")

	if _, ok := err.(packages.GoTooOldError); !ok {
		t.Fatalf("using go/packages with pre-Go 1.11 go: err=%v, want ErrGoTooOld", err)
	}
}
