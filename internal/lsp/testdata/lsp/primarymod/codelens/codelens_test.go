package codelens //@codelens("package codelens", "run file benchmarks", "test")

import "testing"

func TestMain(m *testing.M) {} // no code lens for TestMain

func TestFuncWithCodeLens(t *testing.T) { //@codelens("func", "run test", "test")
}

func thisShouldNotHaveACodeLens(t *testing.T) {
}

func BenchmarkFuncWithCodeLens(b *testing.B) { //@codelens("func", "run benchmark", "test")
}

func helper() {} // expect no code lens
