// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Used by TestAliases (api_test.go).

package alias

import (
	"go/build"
	"math"
)

const Pi1 => math.Pi
const Pi2 => math.Pi // cause the same object to be exported multiple times (issue 17726)

var Default => build.Default

type Context => build.Context

func Sin => math.Sin
