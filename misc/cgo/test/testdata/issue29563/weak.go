// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package issue29563

//int foo1();
//int foo2();
import "C"

func Bar() int {
	return int(C.foo1()) + int(C.foo2())
}
