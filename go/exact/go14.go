// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.4

package exact

import "math/big"

func ratToFloat32(x *big.Rat) (float32, bool) {
	return x.Float32()
}
