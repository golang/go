// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gcimporter

// Temporarily expose version-related functionality so that we can test at
// specific export data versions.

var IExportCommon = iexportCommon

const (
	IExportVersion         = iexportVersion
	IExportVersionGenerics = iexportVersionGenerics
	IExportVersionGo1_18   = iexportVersionGo1_18
)
