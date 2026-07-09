// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specexpr

import (
	"fmt"
	"io"
)

type tracer struct {
	w     io.Writer
	level int
}

func (t *tracer) assign(expr Expr, v Variable, val any, err error) {
	if t == nil {
		return
	}
	fmt.Fprintf(t.w, "%*s├ %s => ", t.level, "", expr)
	if err == nil {
		fmt.Fprintf(t.w, "%s=%v\n", v, val)
	} else {
		fmt.Fprintf(t.w, "%s\n", err)
	}
}

func (t *tracer) check(expr Expr, val any, err error) {
	if t == nil {
		return
	}
	if val == false || err != nil {
		fmt.Fprintf(t.w, "%*s✘ %s", t.level, "", expr)
		if err != nil {
			fmt.Fprintf(t.w, " => %s", err)
		}
		fmt.Fprint(t.w, "\n")
	} else {
		fmt.Fprintf(t.w, "%*s│ %s\n", t.level, "", expr)
	}
}

func (t *tracer) enter(v Variable, val any) {
	if t == nil {
		return
	}
	fmt.Fprintf(t.w, "%*s↳ %s=%v\n", t.level, "", v, val)
	t.level += 4
}

func (t *tracer) exit() {
	if t == nil {
		return
	}
	t.level -= 4
}

func (t *tracer) sat() {
	if t == nil {
		return
	}
	fmt.Fprintf(t.w, "%*s✔ SAT\n", t.level, "")
}
