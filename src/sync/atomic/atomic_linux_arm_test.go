// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package atomic_test

import (
	. "sync/atomic"
	"testing"
)

func TestGeneralCAS64(t *testing.T) {
	testCompareAndSwapUint64(t, GeneralCAS64)
}
