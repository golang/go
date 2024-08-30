// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build compiler_bootstrap

package base

// CompilerBootstrap reports whether the current compiler binary was
// built with -tags=compiler_bootstrap.
const CompilerBootstrap = true
