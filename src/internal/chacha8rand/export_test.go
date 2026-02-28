// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package chacha8rand

var Block = block
var Block_generic = block_generic

func Seed(s *State) [4]uint64 {
	return s.seed
}
