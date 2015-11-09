package buildutil

// This logic was copied from stringsFlag from $GOROOT/src/cmd/go/build.go.

import "fmt"

const TagsFlagDoc = "a list of `build tags` to consider satisfied during the build. " +
	"For more information about build tags, see the description of " +
	"build constraints in the documentation for the go/build package"

// TagsFlag is an implementation of the flag.Value interface that parses
// a flag value in the same manner as go build's -tags flag and
// populates a []string slice.
//
// See $GOROOT/src/go/build/doc.go for description of build tags.
// See $GOROOT/src/cmd/go/doc.go for description of 'go build -tags' flag.
//
// Example:
// 	flag.Var((*buildutil.TagsFlag)(&build.Default.BuildTags), "tags", buildutil.TagsFlagDoc)
type TagsFlag []string

func (v *TagsFlag) Set(s string) error {
	var err error
	*v, err = splitQuotedFields(s)
	if *v == nil {
		*v = []string{}
	}
	return err
}

func splitQuotedFields(s string) ([]string, error) {
	// Split fields allowing '' or "" around elements.
	// Quotes further inside the string do not count.
	var f []string
	for len(s) > 0 {
		for len(s) > 0 && isSpaceByte(s[0]) {
			s = s[1:]
		}
		if len(s) == 0 {
			break
		}
		// Accepted quoted string. No unescaping inside.
		if s[0] == '"' || s[0] == '\'' {
			quote := s[0]
			s = s[1:]
			i := 0
			for i < len(s) && s[i] != quote {
				i++
			}
			if i >= len(s) {
				return nil, fmt.Errorf("unterminated %c string", quote)
			}
			f = append(f, s[:i])
			s = s[i+1:]
			continue
		}
		i := 0
		for i < len(s) && !isSpaceByte(s[i]) {
			i++
		}
		f = append(f, s[:i])
		s = s[i:]
	}
	return f, nil
}

func (v *TagsFlag) String() string {
	return "<tagsFlag>"
}

func isSpaceByte(c byte) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r'
}
