// errorcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f(a, b, c, d ...int)       {} // ERROR "non-final parameter a|only permits one name"
func g(a ...int, b ...int)      {} // ERROR "non-final parameter a|must be last parameter"
func h(...int, ...int, float32) {} // ERROR "non-final parameter|must be last parameter"

type a func(...float32, ...interface{}) // ERROR "non-final parameter|must be last parameter"
type b interface {
	f(...int, ...int)                // ERROR "non-final parameter|must be last parameter"
	g(a ...int, b ...int, c float32) // ERROR "non-final parameter a|must be last parameter"
	valid(...int)
}
