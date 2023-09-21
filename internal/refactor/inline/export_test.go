// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inline

// This file opens back doors for testing.

func (callee *Callee) Effects() []int { return callee.impl.Effects }
