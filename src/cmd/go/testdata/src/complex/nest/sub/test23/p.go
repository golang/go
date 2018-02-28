package test23

// Check that vendor/v3 is used but vendor/v2 is NOT used (sub/vendor/v2 wins).

import (
	"v2"
	"v3"
)

const x = v3.ComplexNestVendorV3
const y = v2.ComplexNestSubVendorV2
