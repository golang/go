// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The following shift tests are disabled in the shared
// testdata/check/shifts.go file because they don't work
// correctly with types2 at the moment. See issue #52080.
// Make sure we keep testing them with go/types.
//
// TODO(gri) Once #52080 is fixed, this file can be
//           deleted in favor of the re-enabled tests
//           in the shared file.

package p

func _() {
	var s uint

	_ = int32(0x80000000 /* ERROR "overflows int32" */ << s)
	// TODO(rfindley) Eliminate the redundant error here.
	_ = int32(( /* ERROR "truncated to int32" */ 0x80000000 /* ERROR "truncated to int32" */ + 0i) << s)

	_ = int(1 + 0i<<0)
	_ = int((1 + 0i) << s)
	_ = int(1.0 << s)
	_ = int(complex(1, 0) << s)
}
