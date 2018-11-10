package standalone_fail_sub_test

import "testing"

func TestThatFails(t *testing.T) {
	t.Run("Sub", func(t *testing.T) {})
	t.Fail()
}
