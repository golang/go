// errorcheck

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 4847: initialization loop is not detected.

package p

type (
	E int
	S int
)

type matcher func(s *S) E

func matchList(s *S) E { return matcher(matchAnyFn)(s) }

var foo = matcher(matchList)

var matchAny = matcher(matchList) // ERROR "initialization loop|depends upon itself"

func matchAnyFn(s *S) (err E) { return matchAny(s) }
