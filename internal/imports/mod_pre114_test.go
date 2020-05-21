package imports

import (
	"testing"

	"golang.org/x/tools/internal/testenv"
)

func TestModVendorAuto_Pre114(t *testing.T) {
	testenv.SkipAfterGo1Point(t, 13)
	testModVendorAuto(t, false)
}
