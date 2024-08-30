// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define X 1

//go:build x // ERROR "misplaced //go:build comment"

