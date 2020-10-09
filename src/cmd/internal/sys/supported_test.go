// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sys

import (
	"internal/testenv"
	"runtime"
	"testing"
)

func TestMustLinkExternalMatchesTestenv(t *testing.T) {
	// MustLinkExternal and testenv.CanInternalLink are the exact opposite.
	if b := MustLinkExternal(runtime.GOOS, runtime.GOARCH); b != !testenv.CanInternalLink() {
		t.Fatalf("MustLinkExternal() == %v, testenv.CanInternalLink() == %v, don't match", b, testenv.CanInternalLink())
	}
}
