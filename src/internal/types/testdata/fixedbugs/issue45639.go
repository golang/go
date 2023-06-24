// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package P

// For now, a lone type parameter is not permitted as RHS in a type declaration (issue #45639).
// // It is not permitted to declare a local type whose underlying
// // type is a type parameters not declared by that type declaration.
// func _[T any]() {
// 	type _ T         // ERROR "cannot use function type parameter T as RHS in type declaration"
// 	type _ [_ any] T // ERROR "cannot use function type parameter T as RHS in type declaration"
// }
