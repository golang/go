// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _[P []byte](p string) {
	_ = (*P)(p /* ERROR "pointer to type parameter" */)
}
