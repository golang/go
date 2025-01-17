// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package A

type (
	_ = A
	A /* ERROR "invalid recursive type: A refers to itself" */ = A
)
