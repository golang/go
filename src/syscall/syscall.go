// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

/*
 * These calls have signatures that are independent of operating system.
 *
 * For simplicity of addressing in assembler, all integers are 64 bits
 * in these calling sequences.
 */

func open(name *byte, mode int64) (ret int64, errno int64);
func close(fd int64) (ret int64, errno int64);
func read(fd int64, buf *byte, nbytes int64) (ret int64, errno int64);
func write(fd int64, buf *byte, nbytes int64) (ret int64, errno int64);

export open, close, read, write
