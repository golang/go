// +build !go1.14

package imports

import (
	"testing"
)

func TestModVendorAuto_Pre114(t *testing.T) {
	testModVendorAuto(t, false)
}
