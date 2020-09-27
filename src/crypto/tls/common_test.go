// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls_test

import (
	"crypto/tls"
	"testing"
)

func TestCloneNilConfig(t *testing.T) {
	var config *tls.Config
	if cc := config.Clone(); cc != nil {
		t.Fatalf("Clone with nil should return nil, got: %+v", cc)
	}
}
