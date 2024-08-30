// errorcheck -t 10

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// The init cycle diagnosis used to take exponential time
// to traverse the call graph paths. This test case takes
// at least two minutes on a modern laptop with the bug
// and runs in a fraction of a second without it.
// 10 seconds (-t 10 above) should be plenty if the code is working.

var x = f() + z() // ERROR "initialization cycle"

func f() int { return a1() + a2() + a3() + a4() + a5() + a6() + a7() }
func z() int { return x }

func a1() int { return b1() + b2() + b3() + b4() + b5() + b6() + b7() }
func a2() int { return b1() + b2() + b3() + b4() + b5() + b6() + b7() }
func a3() int { return b1() + b2() + b3() + b4() + b5() + b6() + b7() }
func a4() int { return b1() + b2() + b3() + b4() + b5() + b6() + b7() }
func a5() int { return b1() + b2() + b3() + b4() + b5() + b6() + b7() }
func a6() int { return b1() + b2() + b3() + b4() + b5() + b6() + b7() }
func a7() int { return b1() + b2() + b3() + b4() + b5() + b6() + b7() }
func a8() int { return b1() + b2() + b3() + b4() + b5() + b6() + b7() }

func b1() int { return a1() + a2() + a3() + a4() + a5() + a6() + a7() }
func b2() int { return a1() + a2() + a3() + a4() + a5() + a6() + a7() }
func b3() int { return a1() + a2() + a3() + a4() + a5() + a6() + a7() }
func b4() int { return a1() + a2() + a3() + a4() + a5() + a6() + a7() }
func b5() int { return a1() + a2() + a3() + a4() + a5() + a6() + a7() }
func b6() int { return a1() + a2() + a3() + a4() + a5() + a6() + a7() }
func b7() int { return a1() + a2() + a3() + a4() + a5() + a6() + a7() }
func b8() int { return a1() + a2() + a3() + a4() + a5() + a6() + a7() }
