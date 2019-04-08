package apidiff

import (
	"bytes"
	"fmt"
	"io"
)

// Report describes the changes detected by Changes.
type Report struct {
	Incompatible, Compatible []string
}

func (r Report) String() string {
	var buf bytes.Buffer
	if err := r.Text(&buf); err != nil {
		return fmt.Sprintf("!!%v", err)
	}
	return buf.String()
}

func (r Report) Text(w io.Writer) error {
	if err := r.TextIncompatible(w, true); err != nil {
		return err
	}
	return r.TextCompatible(w)
}

func (r Report) TextIncompatible(w io.Writer, withHeader bool) error {
	if withHeader {
		return r.writeMessages(w, "Incompatible changes:", r.Incompatible)
	}
	return r.writeMessages(w, "", r.Incompatible)
}

func (r Report) TextCompatible(w io.Writer) error {
	return r.writeMessages(w, "Compatible changes:", r.Compatible)
}

func (r Report) writeMessages(w io.Writer, header string, msgs []string) error {
	if len(msgs) == 0 {
		return nil
	}
	if header != "" {
		if _, err := fmt.Fprintf(w, "%s\n", header); err != nil {
			return err
		}
	}
	for _, m := range msgs {
		if _, err := fmt.Fprintf(w, "- %s\n", m); err != nil {
			return err
		}
	}
	return nil
}
