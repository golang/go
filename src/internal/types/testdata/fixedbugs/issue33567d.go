// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import len "strings"

// An import alias does not change the meaning of a selected package member.
func _() {
	_ = len.Join(3 /* ERRORx "cannot use .* in argument to len.Join$" */, "")
}
