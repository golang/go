// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !darwin
// +build !windows
// +build !freebsd

package runtime

func walltime() (sec int64, nsec int32)
