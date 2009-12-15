// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall


func str(val int) string { // do it here rather than with fmt to avoid dependency
	if val < 0 {
		return "-" + str(-val)
	}
	var buf [32]byte // big enough for int64
	i := len(buf) - 1
	for val >= 10 {
		buf[i] = byte(val%10 + '0')
		i--
		val /= 10
	}
	buf[i] = byte(val + '0')
	return string(buf[i:])
}

func Errstr(errno int) string {
	if errno < 0 || errno >= int(len(errors)) {
		return "error " + str(errno)
	}
	return errors[errno]
}
