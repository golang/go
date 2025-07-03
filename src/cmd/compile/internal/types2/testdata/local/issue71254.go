// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

const (
	B /* ERROR "initialization cycle: B refers to itself" */ = A + B
	A /* ERRORx "initialization cycle for A\\s+.*A refers to B\\s+.*B refers to A" */ = A + B

	C /* ERRORx "initialization cycle for C\\s+.*C refers to D\\s+.*D refers to C" */ = E + D
	D /* ERRORx "initialization cycle for D\\s+.*D refers to C\\s+.*C refers to D" */ = E + C
	E = D + C
)
