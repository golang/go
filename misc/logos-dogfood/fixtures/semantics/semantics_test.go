package semantics

import (
	"slices"
	"sync"
	"testing"
)

// The wide reference arithmetic has no overflow for these inputs. Every
// narrow operand pair is checked, including both signed endpoints and zero.
func TestNarrowArithmetic(t *testing.T) {
	for a := -128; a <= 127; a++ {
		for b := -128; b <= 127; b++ {
			x, y := int8(a), int8(b)
			got := [3]int8{x + y, x - y, x * y}
			want := [3]int8{int8(a + b), int8(a - b), int8(a * b)}
			if got != want {
				t.Fatalf("a=%d b=%d: got %v, want %v", a, b, got, want)
			}
			if b != 0 && x/y != int8(a/b) {
				t.Fatalf("division a=%d b=%d", a, b)
			}
		}
	}
}

func didPanic(f func()) (panicked bool) {
	defer func() { panicked = recover() != nil }()
	f()
	return false
}

func TestSliceCapacityAndPanicOrder(t *testing.T) {
	for n := 0; n <= 33; n++ {
		backing := make([]int, 33)
		xs := backing[:n]
		for high := 0; high <= cap(xs); high++ {
			if got := len(xs[:high:cap(xs)]); got != high {
				t.Fatalf("len=%d high=%d: got %d", n, high, got)
			}
		}
		writes := 0
		if !didPanic(func() {
			for i := 0; i <= n; i++ {
				xs[i] = i + 1
				writes++
			}
		}) || writes != n {
			t.Fatalf("len=%d: panic must follow exactly %d writes, got %d", n, n, writes)
		}
		for i, value := range backing {
			want := 0
			if i < n {
				want = i + 1
			}
			if value != want {
				t.Fatalf("len=%d index=%d: got %d, want %d", n, i, value, want)
			}
		}
	}
}

func TestOverlappingCopyAndAppend(t *testing.T) {
	for n := 0; n <= 33; n++ {
		for shift := -4; shift <= 4; shift++ {
			got := make([]int, n+8)
			for i := range got {
				got[i] = i + 1
			}
			want := slices.Clone(got)
			snapshot := slices.Clone(got[4 : 4+n])
			for i, v := range snapshot {
				want[4+shift+i] = v
			}
			if count := copy(got[4+shift:4+shift+n], got[4:4+n]); count != n || !slices.Equal(got, want) {
				t.Fatalf("copy n=%d shift=%d: got %v, want %v", n, shift, got, want)
			}
		}
	}
	for n := 0; n <= 33; n++ {
		xs := make([]int, n, 2*n)
		for i := range xs {
			xs[i] = i + 1
		}
		xs = append(xs, xs...)
		for i, v := range xs {
			if v != i%n+1 {
				t.Fatalf("append n=%d index=%d: got %d", n, i, v)
			}
		}
	}
}

func TestAssignmentDestinations(t *testing.T) {
	i := 0
	xs := []int{1, 2}
	i, xs[i] = 1, 9
	if i != 1 || !slices.Equal(xs, []int{9, 2}) {
		t.Fatalf("index destination: i=%d xs=%v", i, xs)
	}
	old := xs
	other := []int{3, 4}
	xs, xs[0] = other, 7
	if !slices.Equal(old, []int{7, 2}) || !slices.Equal(xs, []int{3, 4}) {
		t.Fatalf("slice destination: old=%v xs=%v", old, xs)
	}
	m := map[int]int{0: 1, 1: 2}
	k := 0
	k, m[k] = 1, 8
	if k != 1 || m[0] != 8 || m[1] != 2 {
		t.Fatalf("map destination: k=%d m=%v", k, m)
	}
}

func TestRangeFunctionAndClosureLifetime(t *testing.T) {
	stopped := 0
	sequence := func(yield func(int) bool) {
		defer func() { stopped++ }()
		for i := 0; i < 10; i++ {
			if !yield(i) {
				return
			}
		}
	}
	var closures []func() int
	for i := range sequence {
		closures = append(closures, func() int { return i })
		if i == 3 {
			break
		}
	}
	if stopped != 1 || len(closures) != 4 {
		t.Fatalf("iterator cleanup: stopped=%d closures=%d", stopped, len(closures))
	}
	for i, f := range closures {
		if got := f(); got != i {
			t.Fatalf("closure %d: got %d", i, got)
		}
	}
}

type signed interface {
	~int8 | ~int16 | ~int32 | ~int64
}

func sum[T signed](xs []T) T {
	var total T
	for _, x := range xs {
		total += x
	}
	return total
}

func TestGenericNamedIntegers(t *testing.T) {
	type tiny int8
	if got := sum([]tiny{127, 1, -2}); got != 126 {
		t.Fatalf("named integer sum: got %d", got)
	}
	if got := sum([]int64{1<<62 - 1, 1, -(1 << 62)}); got != 0 {
		t.Fatalf("wide integer sum: got %d", got)
	}
}

// The unlock belongs to the mutex, not to the goroutine that acquired it.
// No scheduler timing or sleep is needed to make this protocol terminate.
func TestCrossGoroutineUnlock(t *testing.T) {
	var mu sync.Mutex
	mu.Lock()
	started := make(chan struct{})
	go func() {
		close(started)
		mu.Unlock()
	}()
	<-started
	mu.Lock()
	mu.Unlock()
}
