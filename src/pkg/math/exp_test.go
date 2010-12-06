// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

// Make expGo and exp2Go available for testing.

func ExpGo(x float64) float64  { return expGo(x) }
func Exp2Go(x float64) float64 { return exp2Go(x) }
