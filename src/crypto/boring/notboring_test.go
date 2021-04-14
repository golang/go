// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !boringcrypto

package boring_test

import "testing"

func TestNotBoring(t *testing.T) {
	t.Error("a file tagged !boringcrypto should not build under Go+BoringCrypto")
}
