// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package issue41761a

/*
   typedef struct S41761 S41761;
*/
import "C"

type T struct {
	X *C.S41761
}
