// compile -goexperiment fieldtrack

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func a(x struct{ f int }) { _ = x.f }

func b() { a(struct{ f int }{}) }
