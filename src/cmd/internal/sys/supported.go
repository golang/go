// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sys

// RaceDetectorSupported reports whether goos/goarch supports the race
// detector. There is a copy of this function in cmd/dist/test.go.
func RaceDetectorSupported(goos, goarch string) bool {
	switch goos {
	case "linux":
		return goarch == "amd64" || goarch == "ppc64le" || goarch == "arm64"
	case "darwin", "freebsd", "netbsd", "windows":
		return goarch == "amd64"
	default:
		return false
	}
}

// MSanSupported reports whether goos/goarch supports the memory
// sanitizer option. There is a copy of this function in cmd/dist/test.go.
func MSanSupported(goos, goarch string) bool {
	switch goos {
	case "linux":
		return goarch == "amd64" || goarch == "arm64"
	default:
		return false
	}
}
