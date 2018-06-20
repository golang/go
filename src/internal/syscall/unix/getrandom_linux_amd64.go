// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unix

// Linux getrandom system call number.
// See GetRandom in getrandom_linux.go.
const randomTrap uintptr = 318
