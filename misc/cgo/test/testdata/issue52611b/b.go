// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package issue52611b

/*
typedef struct Bar {
    int X;
} Bar;
*/
import "C"

func GetX2(bar *C.struct_Bar) int32 {
	return int32(bar.X)
}
