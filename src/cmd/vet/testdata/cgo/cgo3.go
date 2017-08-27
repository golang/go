// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Used by TestVetVerbose to test that vet -v doesn't fail because it
// can't find "C".

package testdata

import "C"

func F() {
}
