package standalone_parallel_sub_test

import "testing"

func Test(t *testing.T) {
	ch := make(chan bool, 1)
	t.Run("Sub", func(t *testing.T) {
		t.Parallel()
		<-ch
		t.Run("Nested", func(t *testing.T) {})
	})
	// Ensures that Sub will finish after its t.Run call already returned.
	ch <- true
}
