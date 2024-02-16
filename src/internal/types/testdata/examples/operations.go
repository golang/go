// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// indirection

func _[P any](p P) {
        _ = *p // ERROR "cannot indirect p"
}

func _[P interface{ int }](p P) {
        _ = *p // ERROR "cannot indirect p"
}

func _[P interface{ *int }](p P) {
        _ = *p
}

func _[P interface{ *int | *string }](p P) {
        _ = *p // ERROR "must have identical base types"
}

type intPtr *int

func _[P interface{ *int | intPtr } ](p P) {
        var _ int = *p
}
