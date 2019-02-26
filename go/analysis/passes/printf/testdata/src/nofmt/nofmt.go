package b

import (
	"math/big"
	"testing"
)

func formatBigInt(t *testing.T) {
	t.Logf("%d\n", big.NewInt(4))
}
