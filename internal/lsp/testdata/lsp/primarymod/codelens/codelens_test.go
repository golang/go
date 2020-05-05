package codelens

import "testing"

// no code lens for TestMain
func TestMain(m *testing.M) {
}

func TestFuncWithCodeLens(t *testing.T) { //@ codelens("func", "run test", "test")
}

func thisShouldNotHaveACodeLens(t *testing.T) {
}

func BenchmarkFuncWithCodeLens(b *testing.B) { //@ codelens("func", "run test", "test")
}
