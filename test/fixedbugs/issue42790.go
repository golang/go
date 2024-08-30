// compile

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

const _ = -uint(len(string(1<<32)) - len("\uFFFD"))
