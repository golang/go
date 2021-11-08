// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && linux && !goexperiment.regabiargs

package runtime

import "internal/abi"

func storeRegArgs(dst *sigcontext, src *abi.RegArgs) {
}

func loadRegArgs(dst *abi.RegArgs, src *sigcontext) {
}
