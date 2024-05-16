// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import _ "unsafe"

// As of Go 1.22, the symbols below are found to be pulled via
// linkname in the wild. We provide a push linkname here, to
// keep them accessible with pull linknames.
// This may change in the future. Please do not depend on them
// in new code.

//go:linkname cloneMultipartFileHeader
//go:linkname cloneMultipartForm
//go:linkname cloneOrMakeHeader
//go:linkname cloneTLSConfig
//go:linkname cloneURL
//go:linkname cloneURLValues
//go:linkname newBufioReader
//go:linkname newBufioWriterSize
//go:linkname parseBasicAuth
//go:linkname putBufioReader
//go:linkname putBufioWriter
//go:linkname readRequest

// The compiler doesn't allow linknames on methods, for good reasons.
// We use this trick to push linknames of the methods.
// Do not call them in this package.

//go:linkname badlinkname_serverHandler_ServeHTTP net/http.serverHandler.ServeHTTP
func badlinkname_serverHandler_ServeHTTP(serverHandler, ResponseWriter, *Request)

//go:linkname badlinkname_Transport_Roundtrip net/http.(*Transport).RoundTrip
func badlinkname_Transport_Roundtrip(*Transport, *Request) (*Response, error)
