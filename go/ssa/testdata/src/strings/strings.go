package strings

func Replace(s, old, new string, n int) string
func Index(haystack, needle string) int
func Contains(haystack, needle string) bool
func HasPrefix(s, prefix string) bool
func EqualFold(s, t string) bool
func ToLower(s string) string

type Builder struct{}

func (b *Builder) WriteString(s string) (int, error)
func (b *Builder) String() string
