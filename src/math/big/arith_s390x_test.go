// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build s390x && !math_big_pure_go

package big

import "testing"

func TestNoVec(t *testing.T) {
	// Make sure non-vector versions match vector versions.
	t.Run("AddVV", func(t *testing.T) { testVV(t, "addVV_novec", addVV_novec, addVV) })
	t.Run("SubVV", func(t *testing.T) { testVV(t, "subVV_novec", subVV_novec, subVV) })
}
