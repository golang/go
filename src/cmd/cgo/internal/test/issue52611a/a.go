// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package issue52611a

/*
typedef struct Foo {
    int X;
} Foo;
*/
import "C"

func GetX1(foo *C.struct_Foo) int32 {
	return int32(foo.X)
}
