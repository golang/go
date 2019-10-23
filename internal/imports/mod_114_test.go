// +build go1.14

package imports

import (
	"testing"
)

func TestModVendorAuto_114(t *testing.T) {
	testModVendorAuto(t, true)
}
