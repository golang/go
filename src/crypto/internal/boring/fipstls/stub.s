// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build boringcrypto

// runtime_arg0 is declared in tls.go without a body.
// It's provided by package runtime,
// but the go command doesn't know that.
// Having this assembly file keeps the go command
// from complaining about the missing body
// (because the implementation might be here).
