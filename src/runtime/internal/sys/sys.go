// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// package sys contains system- and configuration- and architecture-specific
// constants used by the runtime.
package sys

// The next line makes 'go generate' write the zgen_*.go files with
// per-OS and per-arch information, including constants
// named goos_$GOOS and goarch_$GOARCH for every
// known GOOS and GOARCH. The constant is 1 on the
// current system, 0 otherwise; multiplying by them is
// useful for defining GOOS- or GOARCH-specific constants.
//go:generate go run gengoos.go
