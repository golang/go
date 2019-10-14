// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

// Export internal functions for testing.
var ExpGo = exp
var Exp2Go = exp2
var HypotGo = hypot
var SqrtGo = sqrt
var TrigReduce = trigReduce

const ReduceThreshold = reduceThreshold
