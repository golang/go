// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// test case from issue

type _ interface{
	m /* ERROR unexpected int in interface type; possibly missing semicolon or newline or } */ int
}

// other cases where the fix for this issue affects the error message

const (
	x int = 10 /* ERROR unexpected literal "foo" in grouped declaration; possibly missing semicolon or newline or \) */ "foo"
)

var _ = []int{1, 2, 3 /* ERROR unexpected int in composite literal; possibly missing comma or } */ int }

type _ struct {
	x y /* ERROR syntax error: unexpected comma in struct type; possibly missing semicolon or newline or } */ ,
}

func f(a, b c /* ERROR unexpected d in parameter list; possibly missing comma or \) */ d) {
	f(a, b, c /* ERROR unexpected d in argument list; possibly missing comma or \) */ d)
}
