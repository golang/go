// errorcheckandrundir -0 -m -d=inlfuncswithclosures=1

//go:build !goexperiment.newinliner

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check correctness of various closure corner cases
// that are expected to be inlined

package ignored
