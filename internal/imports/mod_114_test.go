package imports

import (
	"testing"

	"golang.org/x/tools/internal/testenv"
)

func TestModVendorAuto_114(t *testing.T) {
	testenv.NeedsGo1Point(t, 14)
	testModVendorAuto(t, true)
}
