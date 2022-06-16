// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !(go1.18 && goexperiment.unified)
// +build !go1.18 !goexperiment.unified

package gcimporter

const unifiedIR = false
