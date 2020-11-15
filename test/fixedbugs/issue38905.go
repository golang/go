// compile

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure that literal value can be passed to struct
// blank field with expressions where candiscard(value)
// returns false, see #38905.

package p

type t struct{ _ u }
type u [10]struct{ f int }

func f(x int) t   { return t{u{{1 / x}, {1 % x}}} }
func g(p *int) t  { return t{u{{*p}}} }
func h(s []int) t { return t{u{{s[0]}}} }
