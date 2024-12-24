// Copyright 2024 Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.swissmap

package reflect

func MapGroupOf(x, y Type) Type {
	grp, _ := groupAndSlotOf(x, y)
	return grp
}
