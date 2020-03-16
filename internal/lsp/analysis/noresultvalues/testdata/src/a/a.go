// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noresultvalues

func x() { return nil } // want "no result values expected"

func y() { return nil, "hello" } // want "no result values expected"
