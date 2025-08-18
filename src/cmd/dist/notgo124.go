// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Go 1.26 and later requires Go 1.24.6 as the minimum bootstrap toolchain.
// If cmd/dist is built using an earlier Go version, this file will be
// included in the build and cause an error like:
//
// % GOROOT_BOOTSTRAP=$HOME/sdk/go1.16 ./make.bash
// Building Go cmd/dist using /Users/rsc/sdk/go1.16. (go1.16 darwin/amd64)
// found packages main (build.go) and building_Go_requires_Go_1_24_6_or_later (notgo124.go) in /Users/rsc/go/src/cmd/dist
// %
//
// which is the best we can do under the circumstances.
//
// See go.dev/issue/44505 and go.dev/issue/54265 for more
// background on why Go moved on from Go 1.4 for bootstrap.

//go:build !go1.24

package building_Go_requires_Go_1_24_6_or_later
