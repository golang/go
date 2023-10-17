// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import . "strings"

var _ = Index // use strings

type t struct{ Index int }

var _ = t{Index: 0}
