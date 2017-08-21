package test12

// Check that vendor/v1 is used but vendor/v2 is NOT used (sub/vendor/v2 wins).

import (
	"v1"
	"v2"
)

const x = v1.ComplexNestVendorV1
const y = v2.ComplexNestSubVendorV2
