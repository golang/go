// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func A(arg interface{}) {
	_ = arg.(interface{ Func() int32 })
	_ = arg.(interface{ Func() int32 })
	_ = arg.(interface{ Func() int32 })
	_ = arg.(interface{ Func() int32 })
	_ = arg.(interface{ Func() int32 })
	_ = arg.(interface{ Func() int32 })
	_ = arg.(interface{ Func() int32 })
}
