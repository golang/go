// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package relay sits between main1 and isimd without directly importing "simd".
// It exports V with an unexported SIMD-embedding type from timport/isimd, as
// well as non-SIMD package variables (VdotReduceSum, PTwoP1) initialized from
// isimd's SIMD-dependent package-var chains.
package relay

import "timport/isimd"

// Does NOT directly import simd.
// DOES export a variable with a type, private to timport/isimd, that embeds a simd type.

var V = isimd.F()
var VdotReduceSum = isimd.G()
var PTwoP1 = isimd.H()
