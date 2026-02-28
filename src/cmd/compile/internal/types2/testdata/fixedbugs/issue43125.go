// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

var _ = new(- /* ERROR not a type */ 1)
var _ = new(1 /* ERROR not a type */ + 1)
