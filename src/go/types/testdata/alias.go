// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Used by TestAliases (api_test.go).

package alias

import (
	"go/build"
	"math"
)

const Pi => math.Pi

var Default => build.Default

type Context => build.Context

func Sin => math.Sin
