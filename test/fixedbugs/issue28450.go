// errorcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f(a, b, c, d ...int)       {} // ERROR "non-final parameter a|only permits one name|can only use ... with final parameter"
func g(a ...int, b ...int)      {} // ERROR "non-final parameter a|must be last parameter|can only use ... with final parameter"
func h(...int, ...int, float32) {} // ERROR "non-final parameter|must be last parameter|can only use ... with final parameter"

type a func(...float32, ...interface{}) // ERROR "non-final parameter|must be last parameter|can only use ... with final parameter"
type b interface {
	f(...int, ...int)                // ERROR "non-final parameter|must be last parameter|can only use ... with final parameter"
	g(a ...int, b ...int, c float32) // ERROR "non-final parameter a|must be last parameter|can only use ... with final parameter"
	valid(...int)
}
