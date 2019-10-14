// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package issue30527

import "math"

/*
#include <inttypes.h>

static void issue30527F(char **p, uint64_t mod, uint32_t unused) {}
*/
import "C"

func G(p **C.char) {
	C.issue30527F(p, math.MaxUint64, 1)
	C.issue30527F(p, 1<<64-1, Z)
}
