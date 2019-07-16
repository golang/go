// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package issue26743

// typedef unsigned int uint;
// int C1(uint x) { return x; }
import "C"

var V1 = C.C1(0)
