// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type (
	S  struct{ f int }
	PS *S
)

func a() []*S { return []*S{{f: 1}} }
func b() []PS { return []PS{{f: 1}} }

func c[P *S]() []P { return []P{{f: 1}} }
func d[P PS]() []P { return []P{{f: 1}} }
