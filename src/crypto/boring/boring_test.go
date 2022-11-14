// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build boringcrypto

package boring_test

import (
	"crypto/boring"
	"runtime"
	"testing"
)

func TestEnabled(t *testing.T) {
	supportedPlatform := runtime.GOOS == "linux" && (runtime.GOARCH == "amd64" || runtime.GOARCH == "arm64")
	if supportedPlatform && !boring.Enabled() {
		t.Error("Enabled returned false on a supported platform")
	} else if !supportedPlatform && boring.Enabled() {
		t.Error("Enabled returned true on an unsupported platform")
	}
}
