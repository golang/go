// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package allocator

export func free(*byte)
export func malloc(int) *byte
export func memset(*byte, int, int)
export var footprint int64
export var frozen bool
export func testsizetoclass()
export var allocated int64
export func find(uint64) (obj *byte, size int64, ref *int32, ok bool)
export func gc()
