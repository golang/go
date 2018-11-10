// errorcheck

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that error messages print meaningful values
// for various extreme floating-point constants.

package p

// failure case in issue
const _ int64 = 1e-10000 // ERROR "1e\-10000 truncated"

const (
	_ int64 = 1e10000000 // ERROR "integer too large"
	_ int64 = 1e1000000  // ERROR "integer too large"
	_ int64 = 1e100000   // ERROR "integer too large"
	_ int64 = 1e10000    // ERROR "integer too large"
	_ int64 = 1e1000     // ERROR "integer too large"
	_ int64 = 1e100      // ERROR "10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000 overflows"
	_ int64 = 1e10
	_ int64 = 1e1
	_ int64 = 1e0
	_ int64 = 1e-1       // ERROR "0\.1 truncated"
	_ int64 = 1e-10      // ERROR "1e\-10 truncated"
	_ int64 = 1e-100     // ERROR "1e\-100 truncated"
	_ int64 = 1e-1000    // ERROR "1e\-1000 truncated"
	_ int64 = 1e-10000   // ERROR "1e\-10000 truncated"
	_ int64 = 1e-100000  // ERROR "1e\-100000 truncated"
	_ int64 = 1e-1000000 // ERROR "1e\-1000000 truncated"
)

const (
	_ int64 = -1e10000000 // ERROR "integer too large"
	_ int64 = -1e1000000  // ERROR "integer too large"
	_ int64 = -1e100000   // ERROR "integer too large"
	_ int64 = -1e10000    // ERROR "integer too large"
	_ int64 = -1e1000     // ERROR "integer too large"
	_ int64 = -1e100      // ERROR "\-10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000 overflows"
	_ int64 = -1e10
	_ int64 = -1e1
	_ int64 = -1e0
	_ int64 = -1e-1       // ERROR "\-0\.1 truncated"
	_ int64 = -1e-10      // ERROR "\-1e\-10 truncated"
	_ int64 = -1e-100     // ERROR "\-1e\-100 truncated"
	_ int64 = -1e-1000    // ERROR "\-1e\-1000 truncated"
	_ int64 = -1e-10000   // ERROR "\-1e\-10000 truncated"
	_ int64 = -1e-100000  // ERROR "\-1e\-100000 truncated"
	_ int64 = -1e-1000000 // ERROR "\-1e\-1000000 truncated"
)

const (
	_ int64 = 1.23456789e10000000 // ERROR "integer too large"
	_ int64 = 1.23456789e1000000  // ERROR "integer too large"
	_ int64 = 1.23456789e100000   // ERROR "integer too large"
	_ int64 = 1.23456789e10000    // ERROR "integer too large"
	_ int64 = 1.23456789e1000     // ERROR "integer too large"
	_ int64 = 1.23456789e100      // ERROR "12345678900000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000 overflows"
	_ int64 = 1.23456789e10
	_ int64 = 1.23456789e1        // ERROR "12\.3457 truncated"
	_ int64 = 1.23456789e0        // ERROR "1\.23457 truncated"
	_ int64 = 1.23456789e-1       // ERROR "0\.123457 truncated"
	_ int64 = 1.23456789e-10      // ERROR "1\.23457e\-10 truncated"
	_ int64 = 1.23456789e-100     // ERROR "1\.23457e\-100 truncated"
	_ int64 = 1.23456789e-1000    // ERROR "1\.23457e\-1000 truncated"
	_ int64 = 1.23456789e-10000   // ERROR "1\.23457e\-10000 truncated"
	_ int64 = 1.23456789e-100000  // ERROR "1\.23457e\-100000 truncated"
	_ int64 = 1.23456789e-1000000 // ERROR "1\.23457e\-1000000 truncated"
)

const (
	_ int64 = -1.23456789e10000000 // ERROR "integer too large"
	_ int64 = -1.23456789e1000000  // ERROR "integer too large"
	_ int64 = -1.23456789e100000   // ERROR "integer too large"
	_ int64 = -1.23456789e10000    // ERROR "integer too large"
	_ int64 = -1.23456789e1000     // ERROR "integer too large"
	_ int64 = -1.23456789e100      // ERROR "\-12345678900000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000 overflows"
	_ int64 = -1.23456789e10
	_ int64 = -1.23456789e1        // ERROR "\-12\.3457 truncated"
	_ int64 = -1.23456789e0        // ERROR "\-1\.23457 truncated"
	_ int64 = -1.23456789e-1       // ERROR "\-0\.123457 truncated"
	_ int64 = -1.23456789e-10      // ERROR "\-1\.23457e\-10 truncated"
	_ int64 = -1.23456789e-100     // ERROR "\-1\.23457e\-100 truncated"
	_ int64 = -1.23456789e-1000    // ERROR "\-1\.23457e\-1000 truncated"
	_ int64 = -1.23456789e-10000   // ERROR "\-1\.23457e\-10000 truncated"
	_ int64 = -1.23456789e-100000  // ERROR "\-1\.23457e\-100000 truncated"
	_ int64 = -1.23456789e-1000000 // ERROR "\-1\.23457e\-1000000 truncated"
)
