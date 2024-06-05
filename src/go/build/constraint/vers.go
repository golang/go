// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package constraint

import (
	"strconv"
	"strings"
)

// GoVersion returns the minimum Go version implied by a given build expression.
// If the expression can be satisfied without any Go version tags, GoVersion returns an empty string.
//
// For example:
//
//	GoVersion(linux && go1.22) = "go1.22"
//	GoVersion((linux && go1.22) || (windows && go1.20)) = "go1.20" => go1.20
//	GoVersion(linux) = ""
//	GoVersion(linux || (windows && go1.22)) = ""
//	GoVersion(!go1.22) = ""
//
// GoVersion assumes that any tag or negated tag may independently be true,
// so that its analysis can be purely structural, without SAT solving.
// “Impossible” subexpressions may therefore affect the result.
//
// For example:
//
//	GoVersion((linux && !linux && go1.20) || go1.21) = "go1.20"
func GoVersion(x Expr) string {
	v := minVersion(x, +1)
	if v < 0 {
		return ""
	}
	if v == 0 {
		return "go1"
	}
	return "go1." + strconv.Itoa(v)
}

// minVersion returns the minimum Go major version (9 for go1.9)
// implied by expression z, or if sign < 0, by expression !z.
func minVersion(z Expr, sign int) int {
	switch z := z.(type) {
	default:
		return -1
	case *AndExpr:
		op := andVersion
		if sign < 0 {
			op = orVersion
		}
		return op(minVersion(z.X, sign), minVersion(z.Y, sign))
	case *OrExpr:
		op := orVersion
		if sign < 0 {
			op = andVersion
		}
		return op(minVersion(z.X, sign), minVersion(z.Y, sign))
	case *NotExpr:
		return minVersion(z.X, -sign)
	case *TagExpr:
		if sign < 0 {
			// !foo implies nothing
			return -1
		}
		if z.Tag == "go1" {
			return 0
		}
		_, v, _ := strings.Cut(z.Tag, "go1.")
		n, err := strconv.Atoi(v)
		if err != nil {
			// not a go1.N tag
			return -1
		}
		return n
	}
}

// andVersion returns the minimum Go version
// implied by the AND of two minimum Go versions,
// which is the max of the versions.
func andVersion(x, y int) int {
	if x > y {
		return x
	}
	return y
}

// orVersion returns the minimum Go version
// implied by the OR of two minimum Go versions,
// which is the min of the versions.
func orVersion(x, y int) int {
	if x < y {
		return x
	}
	return y
}
