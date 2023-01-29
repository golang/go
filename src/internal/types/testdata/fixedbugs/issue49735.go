// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _[P1 any, P2 ~byte](s1 P1, s2 P2) {
        _ = append(nil /* ERROR "first argument to append must be a slice; have untyped nil" */ , 0)
        _ = append(s1 /* ERRORx `s1 .* has no core type` */ , 0)
        _ = append(s2 /* ERRORx `s2 .* has core type byte` */ , 0)
}
