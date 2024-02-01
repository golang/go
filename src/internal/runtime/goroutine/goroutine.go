package goroutine

import "strings"

func IsSystemGoroutine(entryFn string) bool {
	// Locked g in extra M (with empty entryFn) is system goroutine.
	return entryFn == "" || entryFn != "runtime.main" && strings.HasPrefix(entryFn, "runtime.")
}
