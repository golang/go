// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check that overlong unions don't bog down type checking.
// Disallow them for now.

package p

type t int

type (
	t00 t; t01 t; t02 t; t03 t; t04 t; t05 t; t06 t; t07 t; t08 t; t09 t
	t10 t; t11 t; t12 t; t13 t; t14 t; t15 t; t16 t; t17 t; t18 t; t19 t
	t20 t; t21 t; t22 t; t23 t; t24 t; t25 t; t26 t; t27 t; t28 t; t29 t
	t30 t; t31 t; t32 t; t33 t; t34 t; t35 t; t36 t; t37 t; t38 t; t39 t
	t40 t; t41 t; t42 t; t43 t; t44 t; t45 t; t46 t; t47 t; t48 t; t49 t
	t50 t; t51 t; t52 t; t53 t; t54 t; t55 t; t56 t; t57 t; t58 t; t59 t
	t60 t; t61 t; t62 t; t63 t; t64 t; t65 t; t66 t; t67 t; t68 t; t69 t
	t70 t; t71 t; t72 t; t73 t; t74 t; t75 t; t76 t; t77 t; t78 t; t79 t
	t80 t; t81 t; t82 t; t83 t; t84 t; t85 t; t86 t; t87 t; t88 t; t89 t
	t90 t; t91 t; t92 t; t93 t; t94 t; t95 t; t96 t; t97 t; t98 t; t99 t
)

type u99 interface {
	t00|t01|t02|t03|t04|t05|t06|t07|t08|t09|
	t10|t11|t12|t13|t14|t15|t16|t17|t18|t19|
	t20|t21|t22|t23|t24|t25|t26|t27|t28|t29|
	t30|t31|t32|t33|t34|t35|t36|t37|t38|t39|
	t40|t41|t42|t43|t44|t45|t46|t47|t48|t49|
	t50|t51|t52|t53|t54|t55|t56|t57|t58|t59|
	t60|t61|t62|t63|t64|t65|t66|t67|t68|t69|
	t70|t71|t72|t73|t74|t75|t76|t77|t78|t79|
	t80|t81|t82|t83|t84|t85|t86|t87|t88|t89|
	t90|t91|t92|t93|t94|t95|t96|t97|t98
}

type u100a interface {
	u99|float32
}

type u100b interface {
	u99|float64
}

type u101 interface {
	t00|t01|t02|t03|t04|t05|t06|t07|t08|t09|
	t10|t11|t12|t13|t14|t15|t16|t17|t18|t19|
	t20|t21|t22|t23|t24|t25|t26|t27|t28|t29|
	t30|t31|t32|t33|t34|t35|t36|t37|t38|t39|
	t40|t41|t42|t43|t44|t45|t46|t47|t48|t49|
	t50|t51|t52|t53|t54|t55|t56|t57|t58|t59|
	t60|t61|t62|t63|t64|t65|t66|t67|t68|t69|
	t70|t71|t72|t73|t74|t75|t76|t77|t78|t79|
	t80|t81|t82|t83|t84|t85|t86|t87|t88|t89|
	t90|t91|t92|t93|t94|t95|t96|t97|t98|t99|
        int // ERROR "cannot handle more than 100 union terms"
}

type u102 interface {
        int /* ERROR "cannot handle more than 100 union terms" */ |string|u100a
}

type u200 interface {
        u100a /* ERROR "cannot handle more than 100 union terms" */ |u100b
}
