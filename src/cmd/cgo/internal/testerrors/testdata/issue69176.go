// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "C"

type T = T // ERROR HERE

//export F
func F(p *T) {}
