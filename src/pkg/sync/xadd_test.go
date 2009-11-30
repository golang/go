// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sync

func Xadd(val *uint32, delta int32) (new uint32) {
	return xadd(val, delta)
}
