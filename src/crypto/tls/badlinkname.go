// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import _ "unsafe"

// As of Go 1.22, the symbols below are found to be pulled via
// linkname in the wild. We provide a push linkname here, to
// keep them accessible with pull linknames.
// This may change in the future. Please do not depend on them
// in new code.

//go:linkname aeadAESGCMTLS13
//go:linkname cipherSuiteTLS13ByID
//go:linkname cipherSuitesTLS13
//go:linkname defaultCipherSuitesTLS13
//go:linkname defaultCipherSuitesTLS13NoAES
//go:linkname errShutdown

// The compiler doesn't allow linknames on methods, for good reasons.
// We use this trick to push linknames of the methods.
// Do not call them in this package.

//go:linkname badlinkname_halfConn_incSeq crypto/tls.(*halfConn).incSeq
func badlinkname_halfConn_incSeq(*halfConn)
