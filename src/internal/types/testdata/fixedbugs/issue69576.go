// -goexperiment=aliastypeparams -gotypesalias=1

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type A[P int] = struct{}

var _ A[string /* ERROR "string does not satisfy int (string missing in int)" */]
