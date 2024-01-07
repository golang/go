// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package unreachable defines an Analyzer that checks for unreachable code.
//
// # Analyzer unreachable
//
// unreachable: check for unreachable code
//
// The unreachable analyzer finds statements that execution can never reach
// because they are preceded by an return statement, a call to panic, an
// infinite loop, or similar constructs.
package unreachable
