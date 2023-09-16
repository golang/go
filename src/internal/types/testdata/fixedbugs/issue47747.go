// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// For now, a lone type parameter is not permitted as RHS in a type declaration (issue #45639).
// type T1[P any] P
// 
// func (T1[_]) m() {}
// 
// func _[P any](x *T1[P]) {
//         // x.m exists because x is of type *T1 where T1 is a defined type
//         // (even though under(T1) is a type parameter)
//         x.m()
// }


func _[P interface{ m() }](x P) {
        x.m()
        // (&x).m doesn't exist because &x is of type *P
        // and pointers to type parameters don't have methods
        (&x).m /* ERROR "type *P is pointer to type parameter, not type parameter" */ ()
}


type T2 interface{ m() }

func _(x *T2) {
        // x.m doesn't exists because x is of type *T2
        // and pointers to interfaces don't have methods
        x.m /* ERROR "type *T2 is pointer to interface, not interface" */()
}

// Test case 1 from issue

type Fooer1[t any] interface {
	Foo(Barer[t])
}
type Barer[t any] interface {
	Bar(t)
}

// For now, a lone type parameter is not permitted as RHS in a type declaration (issue #45639).
// type Foo1[t any] t
// type Bar[t any] t
// 
// func (l Foo1[t]) Foo(v Barer[t]) { v.Bar(t(l)) }
// func (b *Bar[t]) Bar(l t)        { *b = Bar[t](l) }
// 
// func _[t any](f Fooer1[t]) t {
// 	var b Bar[t]
// 	f.Foo(&b)
// 	return t(b)
// }

// Test case 2 from issue

// For now, a lone type parameter is not permitted as RHS in a type declaration (issue #45639).
// type Fooer2[t any] interface {
// 	Foo()
// }
// 
// type Foo2[t any] t
// 
// func (f *Foo2[t]) Foo() {}
// 
// func _[t any](v t) {
// 	var f = Foo2[t](v)
// 	_ = Fooer2[t](&f)
// }
