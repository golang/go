// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !darwin
// +build !windows
// +build !freebsd
// +build !aix
// +build !solaris

package runtime

func walltime1() (sec int64, nsec int32)
