package imports

import (
	"context"
	"testing"

	"golang.org/x/tools/internal/testenv"
)

// Tests that we handle GO111MODULE=on with no go.mod file. See #30855.
func TestNoMainModule(t *testing.T) {
	testenv.NeedsGo1Point(t, 12)
	mt := setup(t, `
-- x.go --
package x
`, "")
	defer mt.cleanup()
	if _, err := mt.env.invokeGo(context.Background(), "mod", "download", "rsc.io/quote@v1.5.1"); err != nil {
		t.Fatal(err)
	}

	mt.assertScanFinds("rsc.io/quote", "quote")
}
