// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _[P struct{ f int }](x P) {
	_ = x.g // ERROR "type P has no field or method g"
}

func _[P struct{ f int } | struct{ g int }](x P) {
	_ = x.g // ERROR "type P has no field or method g"
}
