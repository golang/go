// +build go1.12

package imports

import (
	"testing"
)

// Tests that we handle GO111MODULE=on with no go.mod file. See #30855.
func TestNoMainModule(t *testing.T) {
	mt := setup(t, `
-- x.go --
package x
`, "")
	defer mt.cleanup()
	if _, err := mt.env.invokeGo("mod", "download", "rsc.io/quote@v1.5.1"); err != nil {
		t.Fatal(err)
	}

	mt.assertScanFinds("rsc.io/quote", "quote")
}
