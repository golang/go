// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file is used with build tag timetzdata to embed tzdata into
// the binary.

//go:build timetzdata
// +build timetzdata

package time

import _ "time/tzdata"
