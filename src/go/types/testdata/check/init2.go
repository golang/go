// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// initialization cycles

package init2

// cycles through functions

func f1() int { _ = x1; return 0 }
var x1 /* ERROR initialization cycle */ = f1

func f2() int { _ = x2; return 0 }
var x2 /* ERROR initialization cycle */ = f2()

// cycles through method expressions

type T3 int
func (T3) m() int { _ = x3; return 0 }
var x3 /* ERROR initialization cycle */ = T3.m

type T4 int
func (T4) m() int { _ = x4; return 0 }
var x4 /* ERROR initialization cycle */ = T4.m(0)

type T3p int
func (*T3p) m() int { _ = x3p; return 0 }
var x3p /* ERROR initialization cycle */ = (*T3p).m

type T4p int
func (*T4p) m() int { _ = x4p; return 0 }
var x4p /* ERROR initialization cycle */ = (*T4p).m(nil)

// cycles through method expressions of embedded methods

type T5 struct { E5 }
type E5 int
func (E5) m() int { _ = x5; return 0 }
var x5 /* ERROR initialization cycle */ = T5.m

type T6 struct { E6 }
type E6 int
func (E6) m() int { _ = x6; return 0 }
var x6 /* ERROR initialization cycle */ = T6.m(T6{0})

type T5p struct { E5p }
type E5p int
func (*E5p) m() int { _ = x5p; return 0 }
var x5p /* ERROR initialization cycle */ = (*T5p).m

type T6p struct { E6p }
type E6p int
func (*E6p) m() int { _ = x6p; return 0 }
var x6p /* ERROR initialization cycle */ = (*T6p).m(nil)

// cycles through method values

type T7 int
func (T7) m() int { _ = x7; return 0 }
var x7 /* ERROR initialization cycle */ = T7(0).m

type T8 int
func (T8) m() int { _ = x8; return 0 }
var x8 /* ERROR initialization cycle */ = T8(0).m()

type T7p int
func (*T7p) m() int { _ = x7p; return 0 }
var x7p /* ERROR initialization cycle */ = new(T7p).m

type T8p int
func (*T8p) m() int { _ = x8p; return 0 }
var x8p /* ERROR initialization cycle */ = new(T8p).m()

type T7v int
func (T7v) m() int { _ = x7v; return 0 }
var x7var T7v
var x7v /* ERROR initialization cycle */ = x7var.m

type T8v int
func (T8v) m() int { _ = x8v; return 0 }
var x8var T8v
var x8v /* ERROR initialization cycle */ = x8var.m()

type T7pv int
func (*T7pv) m() int { _ = x7pv; return 0 }
var x7pvar *T7pv
var x7pv /* ERROR initialization cycle */ = x7pvar.m

type T8pv int
func (*T8pv) m() int { _ = x8pv; return 0 }
var x8pvar *T8pv
var x8pv /* ERROR initialization cycle */ = x8pvar.m()

// cycles through method values of embedded methods

type T9 struct { E9 }
type E9 int
func (E9) m() int { _ = x9; return 0 }
var x9 /* ERROR initialization cycle */ = T9{0}.m

type T10 struct { E10 }
type E10 int
func (E10) m() int { _ = x10; return 0 }
var x10 /* ERROR initialization cycle */ = T10{0}.m()

type T9p struct { E9p }
type E9p int
func (*E9p) m() int { _ = x9p; return 0 }
var x9p /* ERROR initialization cycle */ = new(T9p).m

type T10p struct { E10p }
type E10p int
func (*E10p) m() int { _ = x10p; return 0 }
var x10p /* ERROR initialization cycle */ = new(T10p).m()

type T9v struct { E9v }
type E9v int
func (E9v) m() int { _ = x9v; return 0 }
var x9var T9v
var x9v /* ERROR initialization cycle */ = x9var.m

type T10v struct { E10v }
type E10v int
func (E10v) m() int { _ = x10v; return 0 }
var x10var T10v
var x10v /* ERROR initialization cycle */ = x10var.m()

type T9pv struct { E9pv }
type E9pv int
func (*E9pv) m() int { _ = x9pv; return 0 }
var x9pvar *T9pv
var x9pv /* ERROR initialization cycle */ = x9pvar.m

type T10pv struct { E10pv }
type E10pv int
func (*E10pv) m() int { _ = x10pv; return 0 }
var x10pvar *T10pv
var x10pv /* ERROR initialization cycle */ = x10pvar.m()
