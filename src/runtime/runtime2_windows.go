// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

// TODO(brainman): move generation of zsys_windows_*.s out from cmd/dist/buildruntime.c and into here
const cb_max = 2000 // maximum number of windows callbacks allowed (must be in sync with MAXWINCB from cmd/dist/buildruntime.c)
