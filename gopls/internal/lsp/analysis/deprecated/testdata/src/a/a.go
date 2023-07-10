// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package usedeprecated

import "io/ioutil" // want "\"io/ioutil\" is deprecated: .*"

func x() {
	_, _ = ioutil.ReadFile("") // want "ioutil.ReadFile is deprecated: As of Go 1.16, .*"
	Legacy()                   // expect no deprecation notice.
}

// Legacy is deprecated.
//
// Deprecated: use X instead.
func Legacy() {} // want Legacy:"Deprecated: use X instead."
