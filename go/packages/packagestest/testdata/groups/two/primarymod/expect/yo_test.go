package expect_test

import (
	"testdata/groups/two/expect"
	"testing"
)

func TestX(t *testing.T) {
	_ = expect.X //@check("X", "X")
}
