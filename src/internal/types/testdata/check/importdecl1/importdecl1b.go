// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package importdecl1

import . /* ERRORx ".unsafe. imported and not used" */ "unsafe"

type B interface {
	A
}
