// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build loong64 && linux

package cpu

// This is initialized by archauxv and should not be changed after it is
// initialized.
var HWCap uint

func hwcapInit() {
	// TODO: Features that require kernel support like LSX and LASX can
	// be detected here once needed in std library or by the compiler.
}
