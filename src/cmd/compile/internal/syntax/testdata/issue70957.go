// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f() { goto /* ERROR syntax error: unexpected semicolon, expected name */ ;}

func f() { goto } // ERROR syntax error: unexpected }, expected name
