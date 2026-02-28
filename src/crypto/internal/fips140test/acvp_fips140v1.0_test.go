// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build fips140v1.0

package fipstest

import _ "embed"

//go:embed acvp_capabilities_fips140v1.0.json
var capabilitiesJson []byte

var testConfigFile = "acvp_test_fips140v1.0.config.json"
