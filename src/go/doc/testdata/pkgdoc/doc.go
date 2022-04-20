// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pkgdoc

import (
	crand "crypto/rand"
	"math/rand"
)

type T int

type U int

func (T) M() {}

var _ = rand.Int
var _ = crand.Reader
