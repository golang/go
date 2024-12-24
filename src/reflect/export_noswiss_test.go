// Copyright 2024 Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !goexperiment.swissmap

package reflect

import (
	"internal/abi"
	"unsafe"
)

func MapBucketOf(x, y Type) Type {
	return toType(bucketOf(x.common(), y.common()))
}

func CachedBucketOf(m Type) Type {
	t := m.(*rtype)
	if Kind(t.t.Kind_&abi.KindMask) != Map {
		panic("not map")
	}
	tt := (*mapType)(unsafe.Pointer(t))
	return toType(tt.Bucket)
}
