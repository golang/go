package strings

func Replace(s, old, new string, n int) string

func Index(haystack, needle string) int

func Contains(haystack, needle string) bool {
	return Index(haystack, needle) >= 0
}
