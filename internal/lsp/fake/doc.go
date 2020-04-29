// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package fake provides fake implementations of a text editor, LSP client
// plugin, and Sandbox environment for use in tests.
//
// The Editor type provides a high level API for text editor operations
// (open/modify/save/close a buffer, jump to definition, etc.), and the Client
// type exposes an LSP client for the editor that can be connected to a
// language server. By default, the Editor and Client should be compliant with
// the LSP spec: their intended use is to verify server compliance with the
// spec in a variety of environment. Possible future enhancements of these
// types may allow them to misbehave in configurable ways, but that is not
// their primary use.
//
// The Sandbox type provides a facility for executing tests with a temporary
// directory, module proxy, and GOPATH.
package fake
