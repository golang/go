package strings

func Replace(s, old, new string, n int) string

func Index(haystack, needle string) int

func Contains(haystack, needle string) bool {
	return Index(haystack, needle) >= 0
}

func HasPrefix(s, prefix string) bool {
	return len(s) >= len(prefix) && s[0:len(prefix)] == prefix
}

func EqualFold(s, t string) bool
func ToLower(s string) string

type Builder struct {
	s string
}

func (b *Builder) WriteString(s string) (int, error) {
	b.s += s
	return len(s), nil
}
func (b *Builder) String() string { return b.s }
