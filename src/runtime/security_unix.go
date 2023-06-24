// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package runtime

func secure() {
	initSecureMode()

	if !isSecureMode() {
		return
	}

	// When secure mode is enabled, we do two things:
	//   1. ensure the file descriptors 0, 1, and 2 are open, and if not open them,
	//      pointing at /dev/null (or fail)
	//   2. enforce specific environment variable values (currently we only force
	//		GOTRACEBACK=none)
	//
	// Other packages may also disable specific functionality when secure mode
	// is enabled (determined by using linkname to call isSecureMode).
	//
	// NOTE: we may eventually want to enforce (1) regardless of whether secure
	// mode is enabled or not.

	secureFDs()
	secureEnv()
}

func secureEnv() {
	var hasTraceback bool
	for i := 0; i < len(envs); i++ {
		if hasPrefix(envs[i], "GOTRACEBACK=") {
			hasTraceback = true
			envs[i] = "GOTRACEBACK=none"
		}
	}
	if !hasTraceback {
		envs = append(envs, "GOTRACEBACK=none")
	}
}

func secureFDs() {
	const (
		// F_GETFD and EBADF are standard across all unixes, define
		// them here rather than in each of the OS specific files
		F_GETFD = 0x01
		EBADF   = 0x09
	)

	devNull := []byte("/dev/null\x00")
	for i := 0; i < 3; i++ {
		ret, errno := fcntl(int32(i), F_GETFD, 0)
		if ret >= 0 {
			continue
		}
		if errno != EBADF {
			print("runtime: unexpected error while checking standard file descriptor ", i, ", errno=", errno, "\n")
			throw("cannot secure fds")
		}

		if ret := open(&devNull[0], 2 /* O_RDWR */, 0); ret < 0 {
			print("runtime: standard file descriptor ", i, " closed, unable to open /dev/null, errno=", errno, "\n")
			throw("cannot secure fds")
		} else if ret != int32(i) {
			print("runtime: opened unexpected file descriptor ", ret, " when attempting to open ", i, "\n")
			throw("cannot secure fds")
		}
	}
}
