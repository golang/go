// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package issue52611b

import "C"

func GetX1(bar *C.struct_Bar) int32 {
	return int32(bar.X)
}
