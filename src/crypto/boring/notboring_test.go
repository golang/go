// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (goexperiment.boringcrypto && !boringcrypto) || (!goexperiment.boringcrypto && boringcrypto)

package boring_test

import "testing"

func TestNotBoring(t *testing.T) {
	t.Error("goexperiment.boringcrypto and boringcrypto should be equivalent build tags")
}
