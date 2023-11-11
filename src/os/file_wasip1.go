// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build wasip1

package os

import "internal/poll"

// PollFD returns the poll.FD of the file.
//
// Other packages in std that also import internal/poll (such as net)
// can use a type assertion to access this extension method so that
// they can pass the *poll.FD to functions like poll.Splice.
//
// There is an equivalent function in net.rawConn.
//
// PollFD is not intended for use outside the standard library.
func (f *file) PollFD() *poll.FD {
	return &f.pfd
}
